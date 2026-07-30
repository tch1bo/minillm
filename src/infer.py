from contextlib import contextmanager
from pathlib import Path
from typing import Any, Generator, Literal, NamedTuple

import tiktoken
import torch
import tqdm
from pydantic import Field
from pydantic_settings import CliPositionalArg

from src.model import SEPARATOR, Model, ModelLoadArgs
from src.model import Config as ModelConfig
from src.utils import get_logger

logger = get_logger()


@contextmanager
def kv_cache(
    model: Model, batch_size: int, max_len: int, use_fp16: bool
) -> Generator[None, None, None]:
    model.setup_kv_cache(batch_size, max_len, use_fp16)
    try:
        yield
    finally:
        model.delete_kv_cache()


class Generation(NamedTuple):
    output: str
    prob: float


def _generate(
    model: Model,
    tokenizer: tiktoken.Encoding,
    input_ids: torch.Tensor,
    pad_mask: torch.Tensor,
    num_samples: int,
    max_total_len: int,
    temperature: float | None,
    top_p: float | None,
    greedy: bool,
    device: str,
    use_fp16: bool,
    tqdm_desc: str | None = None,
) -> list[list[Generation]]:
    model.eval()
    batch_size, start_num_tokens = input_ids.shape
    input_ids = input_ids.to(device)

    ids = input_ids.repeat_interleave(num_samples, dim=0)
    pad_mask = pad_mask.repeat_interleave(num_samples, dim=0)
    finished = torch.zeros(num_samples * batch_size, dtype=torch.bool, device=device)
    logprobs = torch.zeros(num_samples * batch_size, dtype=torch.float32, device=device)
    gen_lens = torch.zeros(num_samples * batch_size, dtype=torch.long, device=device)

    # preallocate the full pad mask once
    full_pad_mask = torch.zeros(
        (num_samples * batch_size, max_total_len), dtype=torch.bool, device=device
    )
    full_pad_mask[:, :start_num_tokens] = pad_mask

    iterator: Any = range(start_num_tokens, max_total_len)
    if tqdm_desc:
        iterator = tqdm.tqdm(iterator, desc=tqdm_desc)

    with (
        torch.inference_mode(),
        torch.autocast(device, dtype=torch.float16, enabled=use_fp16),
        kv_cache(model, batch_size * num_samples, max_total_len, use_fp16),
    ):

        for i in iterator:
            if i == start_num_tokens:
                # prefill the cache
                hidden = model.forward_no_lm_head(
                    ids,
                    input_pos=torch.arange(start_num_tokens, device=device),
                    pad_mask=full_pad_mask[:, :i],
                )
            else:
                # decode one token at a time
                hidden = model.forward_no_lm_head(
                    ids[..., [i - 1]],
                    input_pos=torch.tensor([i - 1], device=device, dtype=torch.long),
                    pad_mask=full_pad_mask[:, :i],
                )

            logits = model.lm_head(hidden[:, -1, :]).float()
            model_logprobs = logits.log_softmax(dim=-1)

            if temperature is not None:
                logits = logits / temperature

            if top_p is not None:
                sorted_logits, sorted_indices = logits.sort(dim=-1, descending=True)
                cs = sorted_logits.softmax(dim=-1).cumsum(dim=-1)
                mask = cs > top_p
                mask[..., 1:] = mask[..., :-1].clone()
                mask[..., 0] = False
                sorted_logits[mask] = -float("inf")
                logits = torch.gather(sorted_logits, 1, sorted_indices.argsort(-1))

            probs = logits.softmax(dim=-1)
            if greedy:
                next_id = probs.argmax(dim=-1, keepdim=True)
            else:
                next_id = torch.multinomial(probs, num_samples=1)

            # TODO(chibo): we can stop predicting for sequences that are already finished, but that
            # will require some additional bookkeeping
            next_id[finished] = tokenizer.eot_token
            logprobs[~finished] += model_logprobs.gather(1, next_id).reshape(-1)[
                ~finished
            ]
            gen_lens[~finished] += 1
            ids = torch.cat([ids, next_id], dim=1)

            finished |= next_id.squeeze(1) == tokenizer.eot_token
            if finished.all():
                break

    def decode(row: torch.Tensor) -> str:
        tokens = row.reshape(-1)[start_num_tokens:].tolist()
        if tokenizer.eot_token in tokens:
            tokens = tokens[: tokens.index(tokenizer.eot_token)]
        return tokenizer.decode(tokens)

    mean_logprobs = logprobs / gen_lens.clamp(min=1)
    final_probs = mean_logprobs.exp()

    def decode_sample(
        ids_chunk: torch.Tensor, probs_chunk: torch.Tensor
    ) -> list[Generation]:
        return [
            Generation(output=decode(row), prob=p)
            for row, p in zip(ids_chunk, probs_chunk.tolist(), strict=True)
        ]

    return [
        decode_sample(ic, pc)
        for ic, pc in zip(
            ids.chunk(batch_size), final_probs.chunk(batch_size), strict=True
        )
    ]


def generate_top_p(
    model: Model,
    tokenizer: tiktoken.Encoding,
    input_ids: torch.Tensor,
    pad_mask: torch.Tensor,
    num_samples: int,
    max_total_len: int,
    temperature: float,
    top_p: float | None,
    device: str,
    use_fp16: bool,
    tqdm_desc: str | None = None,
) -> list[list[Generation]]:
    result = _generate(
        model,
        tokenizer,
        input_ids,
        pad_mask,
        num_samples=num_samples,
        max_total_len=max_total_len,
        temperature=temperature,
        top_p=top_p,
        greedy=False,
        device=device,
        use_fp16=use_fp16,
        tqdm_desc=tqdm_desc,
    )
    # Sort the responses in decreasing order of probability.
    for i, r in enumerate(result):
        result[i] = sorted(r, key=lambda t: t[1], reverse=True)
    return result


def generate_greedy(
    model: Model,
    tokenizer: tiktoken.Encoding,
    input_ids: torch.Tensor,
    pad_mask: torch.Tensor,
    max_total_len: int,
    device: str,
    use_fp16: bool,
    tqdm_desc: str | None = None,
) -> list[Generation]:
    samples = _generate(
        model,
        tokenizer,
        input_ids,
        pad_mask,
        num_samples=1,
        max_total_len=max_total_len,
        temperature=None,
        top_p=None,
        greedy=True,
        device=device,
        use_fp16=use_fp16,
        tqdm_desc=tqdm_desc,
    )
    for s in samples:
        assert len(s) == 1
    return [s[0] for s in samples]


class InferArgs(ModelLoadArgs):
    text: CliPositionalArg[str]
    model_path: Path
    num_samples: int | None = Field(
        default=None,
        description="None means greedy decoding. An int means top_p sampling with temperature",
    )
    max_tokens: int | None = None
    tokenizer_name: str = "gpt2"
    model_cfg: ModelConfig = Field(default_factory=ModelConfig)
    device: Literal["cuda", "cpu"] = "cpu"
    temp: float = 0.7
    top_p: float = 0.95

    def cli_cmd(self) -> None:
        infer(self)


def infer(args: InferArgs) -> None:
    model, tokenizer = args.load_model_and_tokenizer()

    prompt = args.text + SEPARATOR
    tokens = tokenizer.encode(prompt)
    input_ids = torch.tensor([tokens], device=args.device)
    pad_mask = torch.zeros_like(input_ids, dtype=torch.bool, device=args.device)
    max_len = args.max_tokens if args.max_tokens is not None else args.model_cfg.max_len
    if args.num_samples is None:
        answers = generate_greedy(
            model,
            tokenizer,
            input_ids,
            pad_mask,
            max_len,
            args.device,
            use_fp16=args.device == "cuda",
            tqdm_desc="infering",
        )
    else:
        answers = generate_top_p(
            model,
            tokenizer,
            input_ids,
            pad_mask,
            args.num_samples,
            max_len,
            temperature=args.temp,
            top_p=args.top_p,
            device=args.device,
            use_fp16=args.device == "cuda",
            tqdm_desc="infering",
        )[0]

    for g in answers:
        print(f"prob is: {g.prob}")
        print(g.output)
        print("=" * 120)
