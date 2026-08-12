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
    # NOTE: the output does not include the `eot_token`
    output: str
    prob: float

    # the input (prompt) tokens, with the left padding removed
    input_tokens: torch.Tensor

    # the output tokens, with the right padding removed
    # NOTE: if an `eot_token` was generated, it is included into `output_tokens`
    output_tokens: torch.Tensor

    # True iff the generation was truncated due to `max_len` limit.
    # In other words, True iff no `eot_token` was generated.
    truncated: bool


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
    tqdm_kwargs: dict | None = None,
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
    if tqdm_kwargs is not None:
        iterator = tqdm.tqdm(iterator, **tqdm_kwargs)

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
                    pad_mask=full_pad_mask,
                )
            else:
                # decode one token at a time
                hidden = model.forward_no_lm_head(
                    ids[..., [i - 1]],
                    input_pos=torch.tensor([i - 1], device=device, dtype=torch.long),
                    pad_mask=full_pad_mask,
                )

            logits = model.lm_head(hidden[:, -1, :]).float()
            if model.padded_vocab_size > model.vocab_size:
                # Avoid predicting a non-existing token
                logits[:, model.vocab_size :] = float("-inf")
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

    mean_logprobs = logprobs / gen_lens.clamp(min=1)
    final_probs = mean_logprobs.exp()

    ids, final_probs = ids.to("cpu"), final_probs.to("cpu")

    def make_one_generation(row: torch.Tensor, prob: float) -> Generation:
        all_tokens = row.reshape(-1)

        # Get and unpad the input tokens
        input_tokens = all_tokens[:start_num_tokens]
        input_tokens = input_tokens[input_tokens != tokenizer.eot_token]

        # Get and unpad the output tokens
        output_tokens = all_tokens[start_num_tokens:]
        unpadded_output = output_tokens[output_tokens != tokenizer.eot_token]
        has_eot_token = len(unpadded_output) != len(output_tokens)
        if has_eot_token:
            # At least one eot_token was generated
            output_tokens = output_tokens[: len(unpadded_output) + 1]

        return Generation(
            output=tokenizer.decode(unpadded_output.tolist()),
            prob=prob,
            input_tokens=input_tokens.clone(),
            output_tokens=output_tokens.clone(),
            truncated=not has_eot_token,
        )

    def make_generations_for_sample(
        ids_chunk: torch.Tensor, probs_chunk: torch.Tensor
    ) -> list[Generation]:
        return [
            make_one_generation(row, prob=p)
            for row, p in zip(ids_chunk, probs_chunk.tolist(), strict=True)
        ]

    return [
        make_generations_for_sample(ic, pc)
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
    tqdm_kwargs: dict | None = None,
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
        tqdm_kwargs=tqdm_kwargs,
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
    tqdm_kwargs: dict | None = None,
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
        tqdm_kwargs=tqdm_kwargs,
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
            tqdm_kwargs={"desc": "greedy"},
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
            tqdm_kwargs={"desc": "top_p"},
        )[0]

    for g in answers:
        print(f"prob is: {g.prob}")
        print(g.output)
        print("=" * 120)
