import ast
import multiprocessing
import warnings
from dataclasses import dataclass
from typing import NamedTuple, cast

import tiktoken
import torch
from torch.utils.data import DataLoader

import datasets
from src.model import SEPARATOR
from src.utils import get_logger

logger = get_logger()


class Gsm8kBatch(NamedTuple):
    # (batch_size, padded_length)
    input_ids: torch.Tensor

    # (batch_size, padded_length), a matrix of booleans, True means the token is a padding token
    # the samples in the batch are left-padded
    pad_mask: torch.Tensor

    # (batch_size), a vector of integers, contains the parsed answer to the math problem
    answers: torch.Tensor

    def __len__(self) -> int:
        return self.answers.shape[0]


@dataclass
class Gsm8kCollator:
    tokenizer: tiktoken.Encoding
    sep: list[int]

    def __call__(self, batch: list[dict]) -> Gsm8kBatch:
        tokens = [
            t + self.sep
            for t in self.tokenizer.encode_ordinary_batch(
                [ex["question"] for ex in batch]
            )
        ]

        # shape is (batch_size, max_len_in_batch)
        shape = (len(tokens), max(len(t) for t in tokens))
        input_ids = torch.full(shape, self.tokenizer.eot_token, dtype=torch.long)
        pad_mask = torch.full(shape, True, dtype=torch.bool)

        for i, t in enumerate(tokens):
            input_ids[i, -len(t) :] = torch.tensor(t, dtype=torch.long)
            pad_mask[i, -len(t) :] = False

        answers = torch.tensor(
            [
                # gsm8k uses "," as thousands seprators (e.g. the answer can be "1,080")
                int(ex["answer"].rsplit("#### ", maxsplit=1)[-1].replace(",", ""))
                for ex in batch
            ]
        )
        return Gsm8kBatch(
            input_ids=input_ids,
            pad_mask=pad_mask,
            answers=answers,
        )


def load_gsm8k(
    tokenizer: tiktoken.Encoding,
    train_batch_size: int,
    test_batch_size: int,
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
            batch_size=train_batch_size if is_train else test_batch_size,
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


def _remove_docstring(source: str) -> str:
    """
    Removes the docstring from TinyGSM code snippets.
    We do this because we put the question into the prompt, before any Python code. Having the
    the question inside the docstring (the way it is in the TinyGSM dataset) would redundantly
    duplicate the question and increase the string length unnecessarily.

    Raises on an unexpected (non-TinyGSM) code format.
    """

    # TinyGSM often includes latex-style \$ escapes
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", SyntaxWarning)
        tree = ast.parse(source)
    func_def = tree.body[0]
    assert isinstance(func_def, ast.FunctionDef)
    docstring = func_def.body[0]
    assert isinstance(docstring, ast.Expr)

    lines = source.splitlines(keepends=True)
    new_lines = lines[: docstring.lineno - 1] + lines[docstring.end_lineno :]
    new_source = "".join(new_lines)
    ast.parse(new_source)
    return new_source


@dataclass
class TinyGsmCollator:
    tokenizer: tiktoken.Encoding
    max_len: int
    sep: list[int]

    def __call__(self, batch: list[dict]) -> TinyGsmBatch:
        question = self.tokenizer.encode_ordinary_batch(
            [ex["question"] for ex in batch]
        )
        code = self.tokenizer.encode_ordinary_batch(
            [_remove_docstring(ex["code"]) for ex in batch]
        )

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
    tokenizer: tiktoken.Encoding,
    *,
    batch_size: int,
    max_len: int,
    seed: int,
    start_at_batch: int | None,
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

    # We need to shuffle the dataset directly (as opposed to shuffling inside the DataLoader), to
    # enable instant "start at batch" functionality. Otherwise, we would have to "fast-forward" through
    # the DataLoader, which would involve tokenization.
    # HF caches the result of shuffling.
    filtered_ds = filtered_ds.shuffle(seed=seed).flatten_indices()
    if start_at_batch is not None:
        filtered_ds = filtered_ds.select(
            range(batch_size * start_at_batch, len(filtered_ds))
        )

    return DataLoader(
        cast(torch.utils.data.Dataset, filtered_ds),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=TinyGsmCollator(tokenizer, max_len, sep),
        num_workers=8,
        prefetch_factor=4,
        pin_memory=True,
        drop_last=True,
    )
