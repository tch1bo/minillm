import math
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
from torch.nn.functional import cross_entropy
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR, LRScheduler
from torch.utils.checkpoint import checkpoint
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import datasets
from datasets import Dataset, DatasetDict, load_dataset
from src.model import Config as ModelConfig
from src.model import Model, num_params
from src.utils import get_logger

logger = get_logger()


class PreTrainArgs(BaseModel):
    dataset_dir: Path
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


def save_checkpoint(
    args: PreTrainArgs,
    model: Model,
    optimizer: Optimizer,
    scheduler: LRScheduler,
    scaler: torch.GradScaler,
    step: int,
) -> None:
    out_path = args.out_dir / f"checkpoint_{step:08d}.pt"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    tmp_path = out_path.with_suffix(".tmp")
    state = (
        {
            "step": step,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
        },
    )
    torch.save(state, tmp_path)
    tmp_path.rename(out_path)
    logger.info("saved checkpoint", out_path=out_path)

    if args.max_checkpoints >= 0:
        checkpoints = sorted(out_path.parent.glob("checkpoint_*.pt"))
        for old in checkpoints[: -args.max_checkpoints]:
            logger.info("deleted old checkpoint", path=old)
            old.unlink()


def pre_train(args: PreTrainArgs) -> None:
    torch.manual_seed(args.seed)
    train, test = load_data(args)
    if args.num_eval_batches > 0:
        num_tokens = (
            args.num_eval_batches * args.eval_batch_size + 1
        ) * args.model_cfg.max_len
        test = test.prefix(num_tokens)
        logger.info("reduced the eval dataset", num_eval_tokens=num_tokens)
    logger.info("dataset loaded", train_samples=len(train), test_samples=len(test))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_fp16 = device == "cuda"

    if args.mem_profile_path is not None:
        torch.cuda.memory._record_memory_history(
            enabled="all", context="all", stacks="python", max_entries=1000000
        )

    model = Model(ModelConfig(), vocab_size=args.get_tokenizer().max_token_value + 1)
    model.init_weights()
    model = model.to(device)
    logger.info(
        "model created",
        num_params=num_params(model),
        model=model,
    )

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
    optimizer = torch.optim.AdamW(
        [
            {
                "params": [p for p in model.parameters() if p.dim() >= 2],
                "weight_decay": args.weight_decay,
            },
            {
                # Do not decay layer norms
                "params": [p for p in model.parameters() if p.dim() < 2],
                "weight_decay": 0.0,
            },
        ],
        lr=args.max_lr,
        fused=True,
    )
    total_steps = len(train_dataloader) / args.gradient_acc
    warmup_steps = int(total_steps * args.lr_warmup_ratio)

    def lr_func(step: int) -> float:
        # NOTE: This needs to return a ratio of the max LR (and not the LR directly)

        if step < warmup_steps:
            return (step + 1) / warmup_steps

        min_ratio = args.min_lr / args.max_lr

        r = (step - warmup_steps) / (total_steps - warmup_steps)
        c = (1 + math.cos(r * math.pi)) / 2
        return min_ratio + c * max(1 - min_ratio, 0)

    lr_scheduler = LambdaLR(optimizer, lr_func)
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

    scaler = torch.amp.GradScaler("cuda", enabled=use_fp16)

    def chunked_loss(
        x: torch.Tensor, targets: torch.Tensor, is_train: bool
    ) -> torch.Tensor:
        # NOTE: the direct cross_entropy loss calculation was taking too much memory:
        #   batch_size * len_size * vocab_size * (sizeof(fp32) + sizeof(fp16))
        # which for a batch of 10 was around 3GB
        # Splitting it into N chunks reduces the peak memory N times at a slightly higher compute
        # cost (we have to recompute the `lm_head(hidden)` in the backward pass)
        # This optimization allowed to increase the max batch size from 8 to 16 on my 12GB GPU
        def _chunk_loss(h: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
            logits = model.lm_head(h)
            return cross_entropy(logits.flatten(0, 1), t.flatten(), reduction="sum")

        with torch.autocast(device, dtype=torch.float16, enabled=use_fp16):
            hidden = model.forward_no_lm_head(x)
            loss = hidden.new_zeros((), dtype=torch.float32)
            for h, t in zip(
                hidden.chunk(args.num_ce_chunks, dim=1),
                targets.chunk(args.num_ce_chunks, dim=1),
            ):
                if is_train:
                    loss = loss + checkpoint(_chunk_loss, h, t, use_reentrant=False)
                else:
                    loss = loss + _chunk_loss(h, t)

            return loss / targets.numel()

    running_loss = torch.zeros((), device=device)
    for step, batch in enumerate(tqdm.tqdm(train_dataloader, desc=f"training")):
        model.train()

        x, targets = batch[0].to(device, non_blocking=True), batch[1].to(
            device, non_blocking=True
        )
        loss = chunked_loss(x, targets, is_train=True) / args.gradient_acc
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

                    val_loss += float(chunked_loss(x, targets, is_train=False).item())
            val_time = time.perf_counter() - start_time
            logger.info("finished validation", val_time=f"{val_time:.4f}s")

            writer.add_scalar("val/loss", val_loss / len(test_dataloader), step)
            writer.flush()
            save_checkpoint(
                args,
                model,
                optimizer,
                lr_scheduler,
                scaler,
                step,
            )

        save_checkpoint(
            args,
            model,
            optimizer,
            lr_scheduler,
            scaler,
            step,
        )
