"""
This script is for fine-tuning a model on the TinyGSM dataset.
It uses the gsm8k/test split for testing.
"""

from datetime import datetime
from pathlib import Path
from typing import Self, cast

import tiktoken
import torch
import tqdm
from pydantic import BaseModel, Field, model_validator
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.tensorboard import SummaryWriter

from src.gsm.data import Gsm8kBatch, TinyGsmBatch, load_gsm8k, load_tinygsm
from src.gsm.eval import EvalArgs, Gsm8kGeneration, generate_for_batch
from src.model import (
    Config as ModelConfig,
)
from src.model import (
    Model,
    adam_weight_decay,
    chunked_cross_entropy_loss,
)
from src.utils import (
    get_logger,
    make_cosine_scheduler,
    restore_checkpoint,
    save_checkpoint,
)

logger = get_logger()


class SftArgs(BaseModel):
    seed: int = 42
    tokenizer_name: str = "gpt2"
    # The effective train batch size of 1024 is taken from the TinyGSM paper
    train_batch_size: int = 16
    gradient_acc: int = Field(
        default=64,
        description="the number of batches to accumulate for a gradient update",
    )

    eval_batch_size: int = 128

    # max_lr is taken from the TinyGSM paper
    max_lr: float = 1e-4
    min_lr: float = 1e-5

    # Taken from the TinyGSM paper
    weight_decay: float = 0.01
    out_dir: Path = Field(
        default_factory=lambda: Path(
            f"./out/gsm_sft_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
        )
    )
    save_every: int = 500
    lr_warmup_ratio: float = 0.03
    num_ce_chunks: int = Field(
        default=8,
        description="The number of chunks to split the logits into for computing the cross entropy",
    )
    max_checkpoints: int = Field(
        default=10,
        description="the max number of checkpoints to save. If non-positive, all checkpoints will be saved",
    )
    resume_from_checkpoint: Path | None = None
    pretrained_model_path: Path | None = None
    model_cfg: ModelConfig = Field(default_factory=ModelConfig)

    def cli_cmd(self) -> None:
        run_tinygsm_sft(self)

    @model_validator(mode="after")
    def validate_paths(self) -> Self:
        if self.resume_from_checkpoint is None:
            if self.pretrained_model_path is None:
                raise ValueError(
                    "either --resume_from_checkpoint or --pretrained_model_path must be provided"
                )
            return self

        self.out_dir = self.resume_from_checkpoint.parent
        return self


def run_tinygsm_sft(args: SftArgs) -> None:
    torch.manual_seed(args.seed)
    args.out_dir.mkdir(exist_ok=True, parents=True)
    (args.out_dir / "args.json").write_text(args.model_dump_json(indent=2))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_fp16 = device == "cuda"

    # Load the tokenizer
    tokenizer = tiktoken.get_encoding(args.tokenizer_name)

    # Load the data
    train = load_tinygsm(
        tokenizer,
        batch_size=args.train_batch_size,
        max_len=args.model_cfg.max_len,
        seed=args.seed,
        start_at_batch=0,
    )
    _, test = load_gsm8k(
        tokenizer,
        train_batch_size=1,
        test_batch_size=args.eval_batch_size,
        max_len=args.model_cfg.max_len,
    )

    # Load the model
    model = Model(args.model_cfg, vocab_size=tokenizer.max_token_value + 1).to(device)

    # Init the optimizer and the LR scheduler
    optimizer = adam_weight_decay(model, args.max_lr, args.weight_decay)
    lr_scheduler = LambdaLR(
        optimizer,
        make_cosine_scheduler(
            total_steps=len(train) // args.gradient_acc,
            warmup_ratio=args.lr_warmup_ratio,
            min_lr=args.min_lr,
            max_lr=args.max_lr,
        ),
    )
    scaler = torch.amp.GradScaler("cuda", enabled=use_fp16)

    if args.resume_from_checkpoint is None:
        assert args.pretrained_model_path is not None
        model.load_state_dict(torch.load(args.pretrained_model_path))
        start_step = 0
    else:
        # Load the model
        start_step = (
            restore_checkpoint(
                args.resume_from_checkpoint,
                model,
                optimizer,
                lr_scheduler,
                scaler,
                device,
            )
            + 1
        )

        train = load_tinygsm(
            tokenizer,
            batch_size=args.train_batch_size,
            max_len=args.model_cfg.max_len,
            seed=args.seed,
            start_at_batch=start_step,
        )

    model.compile()

    writer = SummaryWriter(log_dir=args.out_dir)
    logger.info(
        "starting training",
        out_dir=args.out_dir,
        use_fp16=use_fp16,
        effective_batch_size=args.train_batch_size * args.gradient_acc,
    )

    running_loss = torch.zeros((), device=device)
    for i, batch in enumerate(tqdm.tqdm(train, desc="training")):
        step = start_step + i
        model.train()

        batch = cast(TinyGsmBatch, batch)
        x, targets = batch.input_ids.to(device), batch.targets.to(device)

        with torch.autocast(x.device.type, dtype=torch.float16, enabled=use_fp16):
            # We right-pad the batches for fine-tuning on TinyGSM, so technically there are padding
            # tokens, but they don't really matter for the attention computation.
            pad_mask = torch.zeros(
                (x.shape[0], args.model_cfg.max_len), dtype=torch.bool, device=device
            )
            hidden = model.forward_no_lm_head(
                x, input_pos=torch.arange(x.shape[1], device=device), pad_mask=pad_mask
            )
            loss = (
                chunked_cross_entropy_loss(
                    model, hidden, targets, args.num_ce_chunks, is_train=True
                )
                / args.gradient_acc
            )
        running_loss += loss.detach()
        scaler.scale(loss).backward()
        should_save = False

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
            should_save = (grad_update % args.save_every) == 0

        if should_save:
            save_checkpoint(
                args.out_dir,
                model,
                optimizer,
                lr_scheduler,
                scaler,
                step=step,
                max_checkpoints=args.max_checkpoints,
            )

            test_gen: list[Gsm8kGeneration] = []
            with torch.compiler.set_stance("force_eager"):
                # Disable `torch.compile` to avoid recompiling with the kv-cache
                for test_batch in tqdm.tqdm(test, desc="running validation"):
                    test_batch = cast(Gsm8kBatch, test_batch)
                    test_gen.extend(
                        generate_for_batch(
                            model,
                            EvalArgs(
                                model_path=Path("doesnt-matter"),
                                eval_greedy=True,
                                eval_top_p=None,
                                batch_size=args.eval_batch_size,
                            ),
                            tokenizer,
                            test_batch,
                            device,
                        )
                    )
            pass_1 = len(
                [g for g in test_gen if g.greedy and g.answer == g.greedy.e.answer]
            ) / len(test_gen)

            no_answer = len(
                [g for g in test_gen if g.greedy and g.greedy.e.answer is None]
            ) / len(test_gen)
            logger.info(
                "finished the validation step", pass_1=pass_1, no_answer=no_answer
            )
            writer.add_scalar("test/pass_1", pass_1, step)
            writer.add_scalar("test/no_answer", no_answer, step)
            for test_i, test_sample in enumerate(test_gen[:3]):
                if test_sample.greedy is not None:
                    a, g, e = (
                        test_sample.answer,
                        test_sample.greedy.g,
                        test_sample.greedy.e,
                    )
                    writer.add_text(
                        f"test/generation_{test_i}",
                        f"""\
## ground truth: {a}

## generated code
```python
{g.output}
```

## exec result
```json
{e.model_dump_json(indent=2)}
```
""",
                        step,
                    )

    save_checkpoint(
        args.out_dir,
        model,
        optimizer,
        lr_scheduler,
        scaler,
        step=start_step + len(train),
        max_checkpoints=args.max_checkpoints,
    )
    writer.close()
