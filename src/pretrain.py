import math
import multiprocessing
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Literal, assert_never, cast

import numpy as np
import tiktoken
import torch
import tqdm
from pydantic import BaseModel, Field
from torch.nn.functional import cross_entropy
from torch.optim.lr_scheduler import LambdaLR
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
    dataset: Literal["tiny-stories", "fineweb-sample"] = "tiny-stories"
    seed: int = 42
    num_proc: int = Field(default_factory=multiprocessing.cpu_count)
    tokenizer_name: str = "gpt2"
    train_batch_size: int = 64
    eval_batch_size: int = 64
    model_cfg: ModelConfig = Field(default_factory=ModelConfig)
    max_lr: float = 3e-4
    min_lr: float = 3e-5
    weight_decay: float = 0.1
    out_dir: Path = Field(
        default_factory=lambda: Path(
            f"/tmp/minillm/pretrain_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
        )
    )
    save_every: int = 10000
    lr_warmup_ratio: float = 0.1
    mem_profile_path: Path | None = Field(
        default=None,
        description="If not None, then this script will run one forward step with the memory profiler "
        "on and dump the profile to the supplied path. If None, then the pretraining will run without memory profiling",
    )

    def cli_cmd(self) -> None:
        pre_train(self)

    def get_tokenizer(self) -> tiktoken.Encoding:
        return tiktoken.get_encoding(self.tokenizer_name)


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

    @staticmethod
    def from_docs(
        args: PreTrainArgs, ds: datasets.Dataset, mmap_file_name: str
    ) -> "TokenizedDataset":
        tokenizer = args.get_tokenizer()
        logger.info(f"tokenizing", num_strings=len(ds))

        def tokenize(batch) -> dict:
            token_ids = tokenizer.encode_batch(batch["text"])
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

        out_path = args.dataset_dir / mmap_file_name
        logger.info("concatting tokens and mmaping the dataset", mmap_path=out_path)
        total = int(np.sum(tokenized["len"], dtype=np.uint64))
        data = np.memmap(
            out_path,
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
    if args.dataset_dir.is_dir():
        logger.info("loading pretokenized datasets", directory=args.dataset_dir)
        return (
            TokenizedDataset(
                tokens=np.memmap(
                    args.dataset_dir / "train.bin", dtype=np.uint32, mode="r"
                ),
                max_len=args.model_cfg.max_len,
            ),
            TokenizedDataset(
                tokens=np.memmap(
                    args.dataset_dir / "test.bin", dtype=np.uint32, mode="r"
                ),
                max_len=args.model_cfg.max_len,
            ),
        )
    else:
        args.dataset_dir.mkdir(parents=True)

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
        TokenizedDataset.from_docs(args, train, "train.bin"),
        TokenizedDataset.from_docs(args, test, "test.bin"),
    )


def pre_train(args: PreTrainArgs) -> None:
    torch.manual_seed(args.seed)
    train, test = load_data(args)
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
    total_steps = len(train_dataloader)
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
    logger.info("starting training", out_dir=args.out_dir, use_fp16=use_fp16)

    def get_loss(batch: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        x, targets = batch
        x, targets = x.to(device, non_blocking=True), targets.to(
            device, non_blocking=True
        )
        with torch.autocast(device, dtype=torch.float16, enabled=use_fp16):
            logits = model(x)
            return cross_entropy(
                logits.reshape(-1, logits.shape[-1]), targets.reshape(-1)
            )

    scaler = torch.amp.GradScaler("cuda", enabled=use_fp16)
    for i, batch in enumerate(tqdm.tqdm(train_dataloader, desc=f"training")):
        model.train()

        loss = get_loss(batch)
        if args.mem_profile_path is not None:
            torch.cuda.memory._dump_snapshot(str(args.mem_profile_path))
            logger.info(
                "dumped the memory profile after one forward pass",
                out_path=args.mem_profile_path,
            )
            return

        optimizer.zero_grad()
        scaler.scale(loss).backward()

        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scale_before = scaler.get_scale()
        scaler.step(optimizer)
        scaler.update()

        writer.add_scalar("train/loss", loss.item(), i)
        writer.add_scalar("train/lr", lr_scheduler.get_last_lr()[0], i)

        if scaler.get_scale() >= scale_before:
            lr_scheduler.step()

        if i % args.save_every == 0:
            logger.info("running validation", step=i)
            start_time = time.perf_counter()
            model.eval()
            val_loss: float = 0.0
            with torch.inference_mode():
                for b in tqdm.tqdm(test_dataloader, desc="running eval"):
                    loss = get_loss(b)
                    val_loss += float(loss.item())
            val_time = time.perf_counter() - start_time

            writer.add_scalar("val/loss", val_loss / len(test_dataloader), i)
            writer.flush()
            out_path = args.out_dir / f"checkpoint_{i}.pt"
            torch.save(
                {
                    "step": i,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": lr_scheduler.state_dict(),
                    "scaler": scaler.state_dict(),
                },
                out_path,
            )
            logger.info(
                "saved checkpoint", out_path=out_path, val_duration=f"{val_time:.4f}s"
            )
