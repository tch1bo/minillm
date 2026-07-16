import math
from typing import Self, cast

import pydantic
import torch
from torch.nn import Module, ModuleList
from torch.nn.attention import SDPBackend, sdpa_kernel

from src.utils import get_logger

logger = get_logger()


class Config(pydantic.BaseModel):
    num_blocks: int = 12
    model_dim: int = 768
    num_query_heads: int = pydantic.Field(
        default=12, description="The number of query heads in the attention blocks"
    )
    num_kv_heads: int = pydantic.Field(
        default=4,
        description="The number of key and value heads in the attention blocks",
    )
    attention_dim: int = 64
    max_len: int = pydantic.Field(default=1024, description="the max sequence length")
    rope_theta: float = 10000.0
    hidden_dim: int = pydantic.Field(
        default=2048,
        description="the size of the hidden layer of the SWIGLU blocks",
    )

    @pydantic.model_validator(mode="after")
    def check_attention_heads(self) -> Self:
        if self.num_query_heads % self.num_kv_heads != 0:
            raise ValueError("num_query_heads must be a multiple of num_kv_heads")
        return self


class RotaryEmbedding(Module):
    def __init__(self, cfg: Config) -> None:
        super().__init__()

        D = cfg.attention_dim
        freqs = torch.outer(
            torch.arange(0, cfg.max_len, dtype=torch.float32),
            cfg.rope_theta ** (-torch.arange(0, D, 2, dtype=torch.float32) / D),
        )
        self.register_buffer("rope_sin", freqs.sin(), persistent=False)
        self.register_buffer("rope_cos", freqs.cos(), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        num_tokens = x.shape[-2]
        x1 = x[..., ::2]
        x2 = x[..., 1::2]
        sin = self.get_buffer("rope_sin")[:num_tokens]
        cos = self.get_buffer("rope_cos")[:num_tokens]
        y1 = x1 * cos - x2 * sin
        y2 = x1 * sin + x2 * cos
        return torch.stack((y1, y2), dim=-1).flatten(-2)


class SWIGLU(Module):
    def __init__(self, cfg: Config) -> None:
        super().__init__()

        # Fused gate and value matrices
        self.gate_and_value = torch.nn.Linear(
            cfg.model_dim, 2 * cfg.hidden_dim, bias=False
        )
        self.down = torch.nn.Linear(cfg.hidden_dim, cfg.model_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate, value = self.gate_and_value(x).chunk(2, dim=-1)
        gate = torch.nn.functional.silu(gate)
        return self.down(gate * value)


class Attention(Module):
    def __init__(self, cfg: Config, rope: RotaryEmbedding) -> None:
        super().__init__()

        self.Q = torch.nn.Parameter(
            torch.empty(cfg.num_query_heads, cfg.model_dim, cfg.attention_dim)
        )
        self.KV = torch.nn.Parameter(
            torch.empty(2, cfg.num_kv_heads, cfg.model_dim, cfg.attention_dim)
        )
        self.proj = torch.nn.Linear(
            cfg.num_query_heads * cfg.attention_dim, cfg.model_dim, bias=False
        )
        self.rope = rope

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x is (batch, token, model_dim)
        batch_size, num_tokens, model_dim = x.shape

        # self.Q  is    (head, model_dim, attention_dim)
        # self.KV is (2, head, model_dim, attention_dim)
        q = torch.einsum("btd, hda -> bhta", x, self.Q)
        kv = torch.einsum("btd, nhda -> nbhta", x, self.KV)
        k, v = kv.unbind(0)

        # Apply ROPE embeddings
        q = self.rope(q)
        k = self.rope(k)

        # Compute attention
        # NOTE: my GPU is from the older generation and FLASH_ATTENTION is not supported for it.
        # Instead we have to use the EFFICIENT_ATTENTION backend, which is
        # slower, but still reduces the memory consumption from quadratic to linear (of max_len).
        # On a batch of 4, the memory dropped from 2736.0 MB to 218.2 MB
        # (this allowed increasing the batch size from 4 to 10).
        # NOTE(2): An annoying part is that the EFFICIENT_ATTENTION backend
        # doesn't support the `enable_gqa` flag, so we have to repeat the k/v tensors.
        with sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION):
            k = k.repeat_interleave(self.Q.shape[0] // (2 * self.KV.shape[0]), dim=1)
            v = v.repeat_interleave(self.Q.shape[0] // (2 * self.KV.shape[0]), dim=1)
            # z is (b, head, token, attention_dim)
            z = torch.nn.functional.scaled_dot_product_attention(
                q,
                k,
                v,
                is_causal=True,
                dropout_p=0,
                # This should be set to True on a newer GPU (and the interleaving should be dropped)
                enable_gqa=False,
            )
        # reshape z to (b, token, attention_dim * num_heads)
        z = z.transpose(1, 2).reshape(batch_size, num_tokens, -1)
        return self.proj(z)

    def extra_repr(self) -> str:
        return f"Q={tuple(self.Q.shape)} KV={tuple(self.KV.shape)}"


class Block(Module):
    def __init__(self, cfg: Config, rope: RotaryEmbedding) -> None:
        super().__init__()

        self.attention_norm = torch.nn.RMSNorm([cfg.model_dim])
        self.attention = Attention(cfg, rope)
        self.ff_norm = torch.nn.RMSNorm([cfg.model_dim])
        self.ff = SWIGLU(cfg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attention(self.attention_norm(x))
        x = x + self.ff(self.ff_norm(x))
        return x


class Model(Module):
    def __init__(self, cfg: Config, vocab_size: int) -> None:
        super().__init__()

        self.vocab_size = vocab_size
        self.padded_vocab_size = (vocab_size + 63) & ~63

        self.embedding = torch.nn.Embedding(
            num_embeddings=self.padded_vocab_size, embedding_dim=cfg.model_dim
        )
        self.rope = RotaryEmbedding(cfg)
        self.blocks = ModuleList([Block(cfg, self.rope) for _ in range(cfg.num_blocks)])
        self.final_rms_norm = torch.nn.RMSNorm([cfg.model_dim])
        self.lm_head = torch.nn.Linear(
            cfg.model_dim, self.padded_vocab_size, bias=False
        )

    def init_weights(self) -> None:
        std = 0.02
        residual_std = std / math.sqrt(2 * len(self.blocks))

        torch.nn.init.normal_(self.embedding.weight, std=std)
        self.lm_head.weight = self.embedding.weight  # weight tying
        # (if not tying: torch.nn.init.normal_(self.lm_head.weight, std=std))

        for block in self.blocks:
            block = cast(Block, block)
            torch.nn.init.normal_(block.attention.Q, std=std)
            torch.nn.init.normal_(block.attention.KV, std=std)
            torch.nn.init.normal_(block.attention.proj.weight, std=residual_std)
            torch.nn.init.normal_(block.ff.gate_and_value.weight, std=std)
            torch.nn.init.normal_(block.ff.down.weight, std=residual_std)

    def forward_no_lm_head(self, x: torch.Tensor) -> torch.Tensor:
        y = self.embedding(x)
        for block in self.blocks:
            y = block(y)

        return self.final_rms_norm(y)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TODO(chibo): KV-cache for inference?
        y = self.forward_no_lm_head(x)
        return self.lm_head(y)


def num_params(m: Module) -> int:
    return sum(p.numel() for p in m.parameters())
