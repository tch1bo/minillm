import math
from pathlib import Path
from typing import Any, Literal, Mapping, Self, cast, override

import pydantic
import tiktoken
import torch
from torch.nn import Module, ModuleList
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.nn.functional import cross_entropy
from torch.utils.checkpoint import checkpoint

from src.utils import get_logger

logger = get_logger()


# This is used as the separator between the prompt and the response.
SEPARATOR = "\n####\n"


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

    def forward(self, x: torch.Tensor, input_pos: torch.Tensor) -> torch.Tensor:
        x1 = x[..., ::2]
        x2 = x[..., 1::2]
        sin = self.get_buffer("rope_sin").index_select(0, input_pos).to(x.dtype)
        cos = self.get_buffer("rope_cos").index_select(0, input_pos).to(x.dtype)
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
    _k_cache: torch.Tensor | None
    _v_cache: torch.Tensor | None

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

        self.register_buffer("_k_cache", None, persistent=False)
        self.register_buffer("_v_cache", None, persistent=False)

    def forward(
        self, x: torch.Tensor, input_pos: torch.Tensor, pad_mask: torch.Tensor
    ) -> torch.Tensor:
        # x is (batch, token, model_dim)
        batch_size, num_tokens, model_dim = x.shape

        # self.Q  is    (head, model_dim, attention_dim)
        # self.KV is (2, head, model_dim, attention_dim)
        q = torch.einsum("btd, hda -> bhta", x, self.Q)
        kv = torch.einsum("btd, nhda -> nbhta", x, self.KV)

        # k, v are (batch, head, token, attention_dim)
        k, v = kv.unbind(0)

        # Apply ROPE embeddings
        q = self.rope(q, input_pos)
        k = self.rope(k, input_pos)

        # NOTE: The EFFICIENT_ATTENTION backend (see the usage below)
        # doesn't support the `enable_gqa` flag, so we have to repeat the k/v tensors.
        # We do the `repeat_interleave` before writing to the kv-cache so that we don't have to
        # call `repeat_interleave` for the same k/v several times.
        # This results in a bigger cache, but faster compute.
        k = k.repeat_interleave(self.Q.shape[0] // self.KV.shape[1], dim=1)
        v = v.repeat_interleave(self.Q.shape[0] // self.KV.shape[1], dim=1)

        if self._k_cache is not None and self._v_cache is not None:
            # Use the kv-cache
            self._k_cache.index_copy_(2, input_pos, k)
            self._v_cache.index_copy_(2, input_pos, v)

            k = self._k_cache
            v = self._v_cache
        else:
            # Without a cache the source is just the current tokens,
            # so trim the (batch, max_len) pad mask to the (batch, num_tokens) prefix.
            pad_mask = pad_mask[:, : k.shape[2]]

        assert pad_mask.shape == (batch_size, k.shape[2])

        # attn_bias is (batch_size, target_tokens, source_tokens)
        attn_bias = torch.zeros(
            (batch_size, num_tokens, k.shape[2]), dtype=q.dtype, device=q.device
        )
        causal = torch.arange(k.shape[2], device=q.device) <= input_pos.unsqueeze(1)
        attn_bias.masked_fill_(causal.logical_not().unsqueeze(0), float("-inf"))
        attn_bias.masked_fill_(pad_mask.unsqueeze(1), float("-inf"))
        if num_tokens > 1:
            # apply the causal mask
            if not torch.compiler.is_compiling():
                assert (
                    int(input_pos[0]) == 0
                ), "multi-token cache prefill is supported only from the start of the sequence"

            # This is a guard against left-padding. In that case, the first pad token in the sequence
            # would have an all -inf attention row, which would lead to NaNs after softmax.
            # We let the pad tokens attend to themselves and nothing else attends to them anyways.
            diag = torch.arange(num_tokens, device=q.device)
            attn_bias[:, diag, diag] = 0.0

        # Compute attention
        # NOTE: my GPU is from the older generation and FLASH_ATTENTION is not supported for it.
        # Instead we have to use the EFFICIENT_ATTENTION backend, which is
        # slower, but still reduces the memory consumption from quadratic to linear (of max_len).
        # On a batch of 4, the memory dropped from 2736.0 MB to 218.2 MB
        # (this allowed increasing the batch size from 4 to 10).
        with sdpa_kernel(
            [SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH], set_priority=True
        ):
            # z is (b, head, token, attention_dim)
            z = torch.nn.functional.scaled_dot_product_attention(
                q,
                k,
                v,
                # `attn_mask` should be brodcastable to (batch_size, num_heads, target_tokens, source_tokens)
                # before the `unsqueeze` it is: (batch_size, target_tokens, source_tokens)
                attn_mask=attn_bias.unsqueeze(1),
                dropout_p=0,
                enable_gqa=False,
            )
        # reshape z to (b, token, attention_dim * num_heads)
        z = z.transpose(1, 2).reshape(batch_size, num_tokens, -1)
        return self.proj(z)

    def extra_repr(self) -> str:
        return f"Q={tuple(self.Q.shape)} KV={tuple(self.KV.shape)}"

    def setup_kv_cache(self, batch_size: int, max_len: int, use_fp16: bool) -> None:
        # if `enable_gqa` works and the `repeat_interleave` calls are not needed anymore, then
        # the second dimension should become `self.KV.shape[1]`
        shape = (batch_size, self.Q.shape[0], max_len, self.KV.shape[3])
        self.register_buffer(
            "_k_cache",
            torch.zeros(
                shape,
                dtype=torch.float16 if use_fp16 else torch.float32,
                device=self.KV.device,
            ),
            persistent=False,
        )
        self.register_buffer(
            "_v_cache",
            torch.zeros(
                shape,
                dtype=torch.float16 if use_fp16 else torch.float32,
                device=self.KV.device,
            ),
            persistent=False,
        )

    def delete_kv_cache(self) -> None:
        self._k_cache = None
        self._v_cache = None


class Block(Module):
    def __init__(self, cfg: Config, rope: RotaryEmbedding) -> None:
        super().__init__()

        self.attention_norm = torch.nn.RMSNorm([cfg.model_dim])
        self.attention = Attention(cfg, rope)
        self.ff_norm = torch.nn.RMSNorm([cfg.model_dim])
        self.ff = SWIGLU(cfg)

    def forward(
        self, x: torch.Tensor, input_pos: torch.Tensor, pad_mask: torch.Tensor
    ) -> torch.Tensor:
        x = x + self.attention(self.attention_norm(x), input_pos, pad_mask)
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
        torch.nn.init.normal_(self.lm_head.weight, std=std)

        for block in self.blocks:
            block = cast(Block, block)
            torch.nn.init.normal_(block.attention.Q, std=std)
            torch.nn.init.normal_(block.attention.KV, std=std)
            torch.nn.init.normal_(block.attention.proj.weight, std=residual_std)
            torch.nn.init.normal_(block.ff.gate_and_value.weight, std=std)
            torch.nn.init.normal_(block.ff.down.weight, std=residual_std)

    def forward_no_lm_head(
        self,
        x: torch.Tensor,
        # input_pos - the indices (positions) of the `x` tokens in the prompt
        # typically:
        #   for training (wo kv-cache): input_pos = torch.arange(seq_len)
        #   when decoding (w kv-cache): input_pos = torch.tensor([cur_pos])
        # In principle, `input_pos` could simply be `start_pos: int`, but that doesn't play well
        # with `torch.compile`
        input_pos: torch.Tensor,
        # pad_mask is a boolean tensor that indicates which of the tokens are padding tokens
        # (True means "padding token", False - "real token")
        # must be of the size (batch_size, max_len)
        pad_mask: torch.Tensor,
    ) -> torch.Tensor:
        y = self.embedding(x)
        for block in self.blocks:
            y = block(y, input_pos, pad_mask)

        return self.final_rms_norm(y)

    @override
    def load_state_dict(
        self, state_dict: Mapping[str, Any], strict: bool = True, assign: bool = False
    ) -> torch.nn.modules.module._IncompatibleKeys:
        if "model" in state_dict.keys():
            state_dict = state_dict["model"]
        state_dict = {k.removeprefix("_orig_mod."): v for k, v in state_dict.items()}
        return super().load_state_dict(state_dict, strict, assign)

    @staticmethod
    def load_from_file(
        path: Path, cfg: Config, vocab_size: int, device: str
    ) -> "Model":
        d = torch.load(path, map_location="cpu")
        m = Model(cfg, vocab_size)
        if "model" in d.keys():
            d = d["model"]
        m.load_state_dict(d)
        return m.to(device)

    def setup_kv_cache(self, batch_size: int, max_len: int, use_fp16: bool) -> None:
        for module in self.modules():
            if isinstance(module, Attention):
                module.setup_kv_cache(batch_size, max_len, use_fp16)

    def delete_kv_cache(self) -> None:
        for module in self.modules():
            if isinstance(module, Attention):
                module.delete_kv_cache()

    def compile(self, *args, **kwargs) -> None:
        self.forward_no_lm_head = torch.compile(  # type: ignore[method-assign]
            self.forward_no_lm_head, *args, **kwargs
        )


def num_params(m: Module) -> int:
    return sum(p.numel() for p in m.parameters())


class ModelLoadArgs(pydantic.BaseModel):
    model_path: Path
    tokenizer_name: str = "gpt2"
    model_cfg: Config = pydantic.Field(default_factory=Config)
    device: Literal["cuda", "cpu"] = "cuda"

    def load_model_and_tokenizer(self) -> tuple[Model, tiktoken.Encoding]:
        tokenizer = tiktoken.get_encoding(self.tokenizer_name)
        model = Model.load_from_file(
            self.model_path,
            self.model_cfg,
            vocab_size=tokenizer.max_token_value + 1,
            device=self.device,
        )
        model.eval()
        return (model, tokenizer)


def adam_weight_decay(
    model: Model, max_lr: float, weight_decay: float
) -> torch.optim.AdamW:
    return torch.optim.AdamW(
        [
            {
                "params": [p for p in model.parameters() if p.dim() >= 2],
                "weight_decay": weight_decay,
            },
            {
                # Do not decay layer norms
                "params": [p for p in model.parameters() if p.dim() < 2],
                "weight_decay": 0.0,
            },
        ],
        lr=max_lr,
        fused=True,
    )


def chunked_cross_entropy_loss(
    model: Model,
    hidden: torch.Tensor,
    targets: torch.Tensor,
    num_ce_chunks: int,
    *,
    is_train: bool,
) -> torch.Tensor:
    # NOTE: the direct cross_entropy loss calculation was taking too much memory:
    #   batch_size * len_size * vocab_size * (sizeof(fp32) + sizeof(fp16))
    # which for a batch of 10 was around 3GB
    # Splitting it into N chunks reduces the peak memory N times at a slightly higher compute
    # cost (we have to recompute the `lm_head(hidden)` in the backward pass)
    # This optimization allowed to increase the max batch size from 8 to 16 on my 12GB GPU
    def _chunk_loss(h: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        logits = model.lm_head(h)
        return cross_entropy(
            logits.flatten(0, 1), t.flatten(), ignore_index=-100, reduction="sum"
        )

    loss = hidden.new_zeros((), dtype=torch.float32)
    for h, t in zip(
        hidden.chunk(num_ce_chunks, dim=1),
        targets.chunk(num_ce_chunks, dim=1),
    ):
        if is_train:
            loss = loss + checkpoint(_chunk_loss, h, t, use_reentrant=False)
        else:
            loss = loss + _chunk_loss(h, t)

    return loss / (targets != -100).sum().clamp(min=1)
