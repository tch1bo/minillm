from contextlib import contextmanager
from pathlib import Path
from typing import Any, Generator, Literal

import tiktoken
import torch
import tqdm
from pydantic import Field
from pydantic_settings import CliPositionalArg

from src.model import Config as ModelConfig
from src.model import Model, ModelLoadArgs
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


def _generate(
    model: Model,
    tokenizer: tiktoken.Encoding,
    prompt: str | torch.Tensor,
    num_samples: int,
    max_total_len: int,
    temperature: float | None,
    top_p: float | None,
    greedy: bool,
    device: str,
    use_fp16: bool,
    tqdm_desc: str | None = None,
) -> list[tuple[str, float]]:
    model.eval()

    if isinstance(prompt, str):
        input_tokens = tokenizer.encode_ordinary(prompt)
    else:
        input_tokens = prompt.tolist()
    ids = torch.tensor(input_tokens, dtype=torch.long, device=device)

    # ids is (num_samples, len(input_tokens))
    ids = ids.unsqueeze(0).repeat(num_samples, 1)
    finished = torch.zeros(num_samples, dtype=torch.bool, device=device)
    logprobs = torch.zeros(num_samples, dtype=torch.float32, device=device)
    gen_lens = torch.zeros(num_samples, dtype=torch.long, device=device)

    iterator: Any = range(len(input_tokens), max_total_len)
    if tqdm_desc:
        iterator = tqdm.tqdm(iterator, desc=tqdm_desc)

    with (
        torch.inference_mode(),
        torch.autocast(device, dtype=torch.float16, enabled=use_fp16),
        kv_cache(model, num_samples, max_total_len, use_fp16),
    ):

        for i in iterator:
            if i == len(input_tokens):
                # prefill the cache
                hidden = model.forward_no_lm_head(
                    ids, input_pos=torch.arange(len(input_tokens), device=device)
                )
            else:
                # decode one token at a time
                hidden = model.forward_no_lm_head(
                    ids[..., [i - 1]],
                    input_pos=torch.tensor([i - 1], device=device, dtype=torch.long),
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

            # TODO(chibo): instead maybe stop predicting for the finished samples
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
        tokens = row[len(input_tokens) :].tolist()
        if tokenizer.eot_token in tokens:
            tokens = tokens[: tokens.index(tokenizer.eot_token)]
        return tokenizer.decode(tokens)

    mean_logprobs = logprobs / gen_lens.clamp(min=1)
    list_probs = mean_logprobs.exp().tolist()
    return [(decode(row), p) for row, p in zip(ids, list_probs, strict=True)]


def generate_top_p(
    model: Model,
    tokenizer: tiktoken.Encoding,
    prompt: str | torch.Tensor,
    num_samples: int,
    max_total_len: int,
    temperature: float,
    top_p: float | None,
    device: str,
    use_fp16: bool,
    tqdm_desc: str | None = None,
) -> list[tuple[str, float]]:
    return _generate(
        model,
        tokenizer,
        prompt,
        num_samples=num_samples,
        max_total_len=max_total_len,
        temperature=temperature,
        top_p=top_p,
        greedy=False,
        device=device,
        use_fp16=use_fp16,
        tqdm_desc=tqdm_desc,
    )


def generate_greedy(
    model: Model,
    tokenizer: tiktoken.Encoding,
    prompt: str | torch.Tensor,
    max_total_len: int,
    device: str,
    use_fp16: bool,
    tqdm_desc: str | None = None,
) -> tuple[str, float]:
    samples = _generate(
        model,
        tokenizer,
        prompt,
        num_samples=1,
        max_total_len=max_total_len,
        temperature=None,
        top_p=None,
        greedy=True,
        device=device,
        use_fp16=use_fp16,
        tqdm_desc=tqdm_desc,
    )
    assert len(samples) == 1
    return samples[0]


class InferArgs(ModelLoadArgs):
    text: CliPositionalArg[str]
    model_path: Path
    max_tokens: int = 1000
    tokenizer_name: str = "gpt2"
    model_cfg: ModelConfig = Field(default_factory=ModelConfig)
    device: Literal["cuda", "cpu"] = "cpu"

    def cli_cmd(self) -> None:
        infer(self)


def infer(args: InferArgs) -> None:
    # TODO(chibo): change this
    model, tokenizer = args.load_model_and_tokenizer()

    tokens = tokenizer.encode(args.text)
    with torch.inference_mode():
        while len(tokens) < args.max_tokens:
            t = torch.tensor(tokens).reshape((1, -1))
            logits = model.forward(t)[0, -1]
            print(logits.shape)
            smax = logits.softmax(dim=-1)
            top_tokens = smax.argsort(descending=True)[:5]
            print(
                "top 5 tokens: ",
                [(tokenizer.decode([int(t.item())]), smax[t]) for t in top_tokens],
            )
            next_token_id = int(top_tokens[0].item())
            tokens.append(next_token_id)
            print(tokenizer.decode(tokens))
