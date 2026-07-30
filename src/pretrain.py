import multiprocessing
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Literal, Self, assert_never, cast

import numpy as np
import tiktoken
import torch
import tqdm
from pydantic import BaseModel, Field, model_validator
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import datasets
from datasets import Dataset, DatasetDict, load_dataset
from src.model import (
    Config as ModelConfig,
)
from src.model import (
    Model,
    adam_weight_decay,
    chunked_cross_entropy_loss,
    num_params,
)
from src.utils import get_logger, make_cosine_scheduler, save_checkpoint

logger = get_logger()


class PreTrainArgs(BaseModel):
    dataset_dir: Path = Path("./datasets")

    dataset: Literal["tiny-stories", "fineweb-sample"]
    seed: int = 42
    num_proc: int = Field(default_factory=multiprocessing.cpu_count)
    tokenizer_name: str = "gpt2"
    train_batch_size: int = 15
    eval_batch_size: int = 48
    num_eval_batches: int = Field(
        default=0,
        description="the max number of batches from the eval dataset to run validation on. "
        "If non-positive, all batches will be used",
    )
    model_cfg: ModelConfig = Field(default_factory=ModelConfig)
    max_lr: float = 3e-4
    min_lr: float = 3e-5
    weight_decay: float = 0.1
    out_dir: Path = Field(
        default_factory=lambda: Path(
            f"./out/pretrain_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
        )
    )
    save_every: int = 1000
    lr_warmup_ratio: float = 0.1
    mem_profile_path: Path | None = Field(
        default=None,
        description="If not None, then this script will run several steps with the memory profiler "
        "on and dump the profile to the supplied path. If None, then the pretraining will run without memory profiling",
    )
    num_ce_chunks: int = Field(
        default=8,
        description="The number of chunks to split the logits into for computing the cross entropy",
    )
    gradient_acc: int = Field(
        default=32,
        description="the number of batches to accumulate for a gradient update",
    )
    max_checkpoints: int = Field(
        default=10,
        description="the max number of checkpoints to save. If non-positive, all checkpoints will be saved",
    )

    def cli_cmd(self) -> None:
        pre_train(self)

    def get_tokenizer(self) -> tiktoken.Encoding:
        return tiktoken.get_encoding(self.tokenizer_name)

    @model_validator(mode="after")
    def validate_profiler(self) -> Self:
        if self.mem_profile_path is not None:
            logger.warning(
                "setting --gradient_acc to 1, because memory profiling is enabled"
            )
            self.gradient_acc = 1
        return self

    def dataset_path(self) -> Path:
        return self.dataset_dir / f"{self.dataset}-{self.tokenizer_name}"

    def train_bin_path(self) -> Path:
        return self.dataset_path() / "train.bin"

    def test_bin_path(self) -> Path:
        return self.dataset_path() / "test.bin"


@dataclass(kw_only=True, frozen=True)
class TokenizedDataset(torch.utils.data.Dataset):
    tokens: np.ndarray
    max_len: int

    def __len__(self):
        return len(self.tokens) // self.max_len - 1

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor]:
        start = i * self.max_len
        x = torch.tensor(self.tokens[start : start + self.max_len], dtype=torch.long)
        y = torch.tensor(
            self.tokens[start + 1 : start + self.max_len + 1], dtype=torch.long
        )
        return x, y

    def prefix(self, num_tokens: int) -> "TokenizedDataset":
        if num_tokens <= 0:
            return self

        return TokenizedDataset(tokens=self.tokens[:num_tokens], max_len=self.max_len)

    @staticmethod
    def from_docs(
        args: PreTrainArgs, ds: datasets.Dataset, mmap_path: Path
    ) -> "TokenizedDataset":
        tokenizer = args.get_tokenizer()
        logger.info(f"tokenizing", num_strings=len(ds))

        def tokenize(batch) -> dict:
            token_ids = tokenizer.encode_batch(batch["text"], disallowed_special=())
            for t in token_ids:
                t.append(tokenizer.eot_token)
            return {"token_ids": token_ids, "len": [len(ts) for ts in token_ids]}

        tokenized = ds.map(
            tokenize,
            batched=True,
            batch_size=1000,
            num_proc=args.num_proc,
            remove_columns=ds.column_names,
            # keep_in_memory=True,
        )

        logger.info("concatting tokens and mmaping the dataset", mmap_path=mmap_path)
        mmap_path.parent.mkdir(exist_ok=True, parents=True)
        total = int(np.sum(tokenized["len"], dtype=np.uint64))
        data = np.memmap(
            mmap_path,
            dtype=np.uint32,
            mode="w+",
            shape=(total,),
        )
        pos, NUM_SHARDS = 0, 512
        for shard_idx in tqdm.trange(NUM_SHARDS):
            shard = tokenized.shard(
                num_shards=NUM_SHARDS, index=shard_idx, contiguous=True
            ).with_format("numpy")
            chunk = np.concatenate(shard["token_ids"])
            data[pos : pos + len(chunk)] = chunk
            pos += len(chunk)
        data.flush()
        return TokenizedDataset(tokens=data, max_len=args.model_cfg.max_len)


def load_data(args: PreTrainArgs) -> tuple[TokenizedDataset, TokenizedDataset]:
    train_path, test_path = args.train_bin_path(), args.test_bin_path()
    if train_path.is_file() and test_path.is_file():
        logger.info("loading pretokenized datasets", train=train_path, test=test_path)
        return (
            TokenizedDataset(
                tokens=np.memmap(train_path, dtype=np.uint32, mode="r"),
                max_len=args.model_cfg.max_len,
            ),
            TokenizedDataset(
                tokens=np.memmap(test_path, dtype=np.uint32, mode="r"),
                max_len=args.model_cfg.max_len,
            ),
        )

    # use name="sample-10BT" to use the 10BT sample
    match args.dataset:
        case "tiny-stories":
            ds = cast(DatasetDict, load_dataset("roneneldan/TinyStories"))
            train, test = ds["train"], ds["validation"]

        case "fineweb-sample":
            ds = cast(
                DatasetDict,
                cast(
                    Dataset,
                    load_dataset(
                        "HuggingFaceFW/fineweb-edu",
                        name="sample-10BT",
                        # NOTE: there's no test split for sample-10BT
                        split="train",
                    ),
                ).train_test_split(test_size=0.1, seed=args.seed),
            )
            train, test = ds["train"], ds["test"]
        case _:
            assert_never(args.dataset)

    return (
        TokenizedDataset.from_docs(args, train, train_path),
        TokenizedDataset.from_docs(args, test, test_path),
    )


def pre_train(args: PreTrainArgs) -> None:
    torch.manual_seed(args.seed)

    # Load the data
    train, test = load_data(args)
    if args.num_eval_batches > 0:
        num_tokens = (
            args.num_eval_batches * args.eval_batch_size + 1
        ) * args.model_cfg.max_len
        test = test.prefix(num_tokens)
        logger.info("reduced the eval dataset", num_eval_tokens=num_tokens)
    train_dataloader = DataLoader(
        train,
        batch_size=args.train_batch_size,
        shuffle=True,
        pin_memory=True,
        drop_last=True,
    )
    test_dataloader = DataLoader(
        test, batch_size=args.eval_batch_size, drop_last=True, pin_memory=True
    )
    logger.info("dataset loaded", train_samples=len(train), test_samples=len(test))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_fp16 = device == "cuda"

    if args.mem_profile_path is not None:
        torch.cuda.memory._record_memory_history(
            enabled="all", context="all", stacks="python", max_entries=1000000
        )

    # Init the model
    model = Model(ModelConfig(), vocab_size=args.get_tokenizer().max_token_value + 1)
    model.init_weights()
    model = model.to(device)
    model.compile()
    logger.info(
        "model created",
        num_params=num_params(model),
        model=model,
    )

    # Init the optimizer and the LR scheduler
    optimizer = adam_weight_decay(model, args.max_lr, args.weight_decay)
    lr_scheduler = LambdaLR(
        optimizer,
        make_cosine_scheduler(
            total_steps=len(train_dataloader) // args.gradient_acc,
            warmup_ratio=args.lr_warmup_ratio,
            min_lr=args.min_lr,
            max_lr=args.max_lr,
        ),
    )
    scaler = torch.amp.GradScaler("cuda", enabled=use_fp16)

    args.out_dir.mkdir(exist_ok=True, parents=True)
    (args.out_dir / "args.json").write_text(args.model_dump_json(indent=2))
    writer = SummaryWriter(log_dir=args.out_dir)
    logger.info(
        "starting training",
        out_dir=args.out_dir,
        use_fp16=use_fp16,
        tokens_per_grad_update=args.train_batch_size
        * args.gradient_acc
        * args.model_cfg.max_len,
    )

    running_loss = torch.zeros((), device=device)
    for step, batch in enumerate(tqdm.tqdm(train_dataloader, desc="training")):
        model.train()

        x, targets = batch[0].to(device, non_blocking=True), batch[1].to(
            device, non_blocking=True
        )

        with torch.autocast(x.device.type, dtype=torch.float16, enabled=use_fp16):
            pad_mask = torch.zeros(x.shape, dtype=torch.bool, device=device)
            hidden = model.forward_no_lm_head(
                x, input_pos=torch.arange(x.shape[1], device=device), pad_mask=pad_mask
            )
            loss = (
                chunked_cross_entropy_loss(
                    model,
                    hidden,
                    targets,
                    args.num_ce_chunks,
                    is_train=True,
                )
                / args.gradient_acc
            )
        running_loss += loss.detach()
        scaler.scale(loss).backward()
        should_run_eval = False

        if (step + 1) % args.gradient_acc == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scale_before = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            writer.add_scalar("train/loss", running_loss.item(), step)
            writer.add_scalar("train/lr", lr_scheduler.get_last_lr()[0], step)

            if scaler.get_scale() >= scale_before:
                lr_scheduler.step()
            optimizer.zero_grad()
            running_loss.zero_()

            grad_update = (step + 1) // args.gradient_acc
            should_run_eval = (
                args.mem_profile_path is None and grad_update % args.save_every == 0
            )

        if args.mem_profile_path is not None and step == 2:
            torch.cuda.memory._dump_snapshot(str(args.mem_profile_path))
            logger.info(
                "dumped the memory profile after one forward pass",
                out_path=args.mem_profile_path,
            )
            return

        if should_run_eval:
            logger.info("running validation", step=step)
            start_time = time.perf_counter()
            model.eval()
            val_loss: float = 0.0
            with (
                torch.inference_mode(),
                torch.autocast(device, dtype=torch.float16, enabled=use_fp16),
            ):
                for b in tqdm.tqdm(test_dataloader, desc="running eval"):
                    x, targets = b[0].to(device, non_blocking=True), b[1].to(
                        device, non_blocking=True
                    )
                    pad_mask = torch.zeros(x.shape, dtype=torch.bool, device=device)
                    hidden = model.forward_no_lm_head(
                        x,
                        input_pos=torch.arange(x.shape[1], device=device),
                        pad_mask=pad_mask,
                    )
                    loss = chunked_cross_entropy_loss(
                        model,
                        hidden,
                        targets,
                        args.num_ce_chunks,
                        is_train=False,
                    )

                    val_loss += float(loss.item())
            val_time = time.perf_counter() - start_time
            logger.info("finished validation", val_time=f"{val_time:.4f}s")

            writer.add_scalar("val/loss", val_loss / len(test_dataloader), step)
            writer.flush()
            save_checkpoint(
                args.out_dir,
                model,
                optimizer,
                lr_scheduler,
                scaler,
                step=step,
                max_checkpoints=args.max_checkpoints,
            )

    save_checkpoint(
        args.out_dir,
        model,
        optimizer,
        lr_scheduler,
        scaler,
        step=len(train_dataloader),
        max_checkpoints=args.max_checkpoints,
    )
    writer.close()
