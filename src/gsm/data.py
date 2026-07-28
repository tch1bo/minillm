import multiprocessing
from dataclasses import dataclass
from typing import NamedTuple, cast

import tiktoken
import torch
from torch.utils.data import DataLoader

import datasets
from src.utils import get_logger

logger = get_logger()

SEPARATOR = "\n####\n"


class Gsm8kSample(NamedTuple):
    input_ids: torch.Tensor  # a 1d vector of tokens
    answer: int


@dataclass
class Gsm8kCollator:
    tokenizer: tiktoken.Encoding
    sep: list[int]

    def __call__(self, batch: list[dict]) -> Gsm8kSample:
        # NOTE: This collator currently works only with batches of size 1
        # This is done to avoid implementing left-side padding and support
        # of attention masks in the model.
        assert len(batch) == 1
        q = batch[0]["question"]
        tokens = self.tokenizer.encode_ordinary(q) + self.sep

        input_ids = torch.tensor(tokens, dtype=torch.long)

        # gsm8k uses "," as thousands seprators (e.g. the answer can be "1,080")
        answer = int(
            batch[0]["answer"].rsplit("#### ", maxsplit=1)[-1].replace(",", "")
        )
        return Gsm8kSample(input_ids=input_ids, answer=answer)


def load_gsm8k(
    tokenizer: tiktoken.Encoding,
    max_len: int,
) -> tuple[DataLoader, DataLoader]:
    gsm8k = datasets.load_dataset("openai/gsm8k", "main")
    sep = tokenizer.encode_ordinary(SEPARATOR)

    train, test = gsm8k["train"], gsm8k["test"]

    def good_length(batch: dict[str, list]) -> list[bool]:
        q_lens = [len(t) for t in tokenizer.encode_ordinary_batch(batch["question"])]
        return [q + len(sep) <= max_len for q in q_lens]

    def _make_dataloader(is_train: bool) -> DataLoader:
        ds = train if is_train else test
        # HF caches the result of filtering
        filtered_ds = ds.filter(
            good_length,
            batched=True,
            batch_size=1000,
            num_proc=multiprocessing.cpu_count(),
            desc="filtering long samples",
        )
        logger.info(
            "done removing long samples",
            ds_size_before=len(ds),
            ds_size_after=len(filtered_ds),
            split="train" if is_train else "test",
        )
        return DataLoader(
            cast(torch.utils.data.Dataset, filtered_ds),
            batch_size=1,
            shuffle=is_train,
            collate_fn=Gsm8kCollator(tokenizer, sep),
            num_workers=4,
            pin_memory=True,
        )

    return (_make_dataloader(True), _make_dataloader(False))


class TinyGsmBatch(NamedTuple):
    input_ids: torch.Tensor  # (batch_size, padded_length)

    # (batch_size, padded_length)
    # a value of -100 should be ignored in the loss computation
    targets: torch.Tensor


@dataclass
class TinyGsmCollator:
    tokenizer: tiktoken.Encoding
    max_len: int
    sep: list[int]

    def __call__(self, batch: list[dict]) -> TinyGsmBatch:
        question = self.tokenizer.encode_ordinary_batch(
            [ex["question"] for ex in batch]
        )
        code = self.tokenizer.encode_ordinary_batch([ex["code"] for ex in batch])

        tokens = []
        for i in range(len(batch)):
            tokens.append(question[i] + self.sep + code[i] + [self.tokenizer.eot_token])

        # shape is (batch_size, max_len_in_batch)
        shape = (len(tokens), max(len(t) for t in tokens))
        input_ids = torch.full(shape, self.tokenizer.eot_token, dtype=torch.long)
        targets = torch.full(shape, -100, dtype=torch.long)

        for i, t in enumerate(tokens):
            input_ids[i, : len(t)] = torch.tensor(t, dtype=torch.long)

            input_len = len(question[i]) + len(self.sep)
            targets[i, input_len - 1 : len(t) - 1] = torch.tensor(
                t[input_len:], dtype=torch.long
            )

        return TinyGsmBatch(input_ids=input_ids, targets=targets)


def load_tinygsm(
    tokenizer: tiktoken.Encoding, batch_size: int, max_len: int
) -> DataLoader:
    ds = datasets.load_dataset("TinyGSM/TinyGSM", split="train")
    sep = tokenizer.encode_ordinary(SEPARATOR)

    def good_length(batch: dict[str, list]) -> list[bool]:
        q_lens = [len(t) for t in tokenizer.encode_ordinary_batch(batch["question"])]
        c_lens = [len(t) for t in tokenizer.encode_ordinary_batch(batch["code"])]
        return [q + len(sep) + c + 1 <= max_len for q, c in zip(q_lens, c_lens)]

    # HF caches the result of filtering
    filtered_ds = ds.filter(
        good_length,
        batched=True,
        batch_size=1000,
        num_proc=multiprocessing.cpu_count(),
        desc="filtering long samples",
    )
    logger.info(
        "done removing long samples",
        ds_size_before=len(ds),
        ds_size_after=len(filtered_ds),
    )
    return DataLoader(
        cast(torch.utils.data.Dataset, filtered_ds),
        batch_size=batch_size,
        shuffle=True,
        collate_fn=TinyGsmCollator(tokenizer, max_len, sep),
        num_workers=4,
        pin_memory=True,
        drop_last=True,
    )
