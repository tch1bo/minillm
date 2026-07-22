from pathlib import Path
from typing import Literal

import tiktoken
import torch
from pydantic import BaseModel, Field
from pydantic_settings import CliPositionalArg

from src.model import Config as ModelConfig
from src.model import Model
from src.utils import get_logger

logger = get_logger()


class InferArgs(BaseModel):
    text: CliPositionalArg[str]
    model_path: Path
    max_tokens: int = 1000
    tokenizer_name: str = "gpt2"
    model_cfg: ModelConfig = Field(default_factory=ModelConfig)
    device: Literal["cuda", "cpu"] = "cpu"

    def cli_cmd(self) -> None:
        infer(self)


def infer(args: InferArgs) -> None:
    tokenizer = tiktoken.get_encoding(args.tokenizer_name)
    model = Model.load_from_file(
        args.model_path,
        args.model_cfg,
        vocab_size=tokenizer.max_token_value + 1,
        device=args.device,
    )
    model.eval()

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
