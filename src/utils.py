import datetime
import math
from pathlib import Path
from typing import Callable

import structlog
import torch
from tqdm import tqdm


def _add_timestamp(logger, method, event_dict):
    now = datetime.datetime.now()
    event_dict["timestamp"] = (
        now.strftime("%H:%M:%S.") + f"{now.microsecond // 10000:02d}"
    )
    return event_dict


class _TqdmWriteFile:
    def write(self, msg):
        tqdm.write(msg, end="")

    def flush(self):
        pass


structlog.configure(
    processors=[
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        _add_timestamp,
        structlog.processors.StackInfoRenderer(),
        structlog.dev.ConsoleRenderer(),
    ],
    wrapper_class=structlog.make_filtering_bound_logger(0),
    context_class=dict,
    logger_factory=structlog.PrintLoggerFactory(file=_TqdmWriteFile()),  # type: ignore
    cache_logger_on_first_use=True,
)


def get_logger(**kwargs: object) -> structlog.stdlib.BoundLogger:
    return structlog.get_logger(**kwargs)


logger = get_logger()


def make_cosine_scheduler(
    *, total_steps: int, warmup_ratio: float, min_lr: float, max_lr: float
) -> Callable[[int], float]:
    def cosine_lr(step: int) -> float:
        warmup_steps = int(total_steps * warmup_ratio)
        # NOTE: This needs to return a ratio of the max LR (and not the LR directly)

        if step < warmup_steps:
            return (step + 1) / warmup_steps

        min_ratio = min_lr / max_lr

        r = (step - warmup_steps) / (total_steps - warmup_steps)
        c = (1 + math.cos(r * math.pi)) / 2
        return min_ratio + c * max(1 - min_ratio, 0)

    return cosine_lr


def save_checkpoint(
    out_dir: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: torch.GradScaler,
    *,
    step: int,
    max_checkpoints: int,
) -> None:
    out_path = out_dir / f"checkpoint_{step + 1:08d}.pt"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    tmp_path = out_path.with_suffix(".tmp")
    state = {
        "step": step,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict(),
    }
    torch.save(state, tmp_path)
    tmp_path.rename(out_path)
    logger.info("saved checkpoint", out_path=out_path)

    if max_checkpoints >= 0:
        checkpoints = sorted(out_path.parent.glob("checkpoint_*.pt"))
        for old in checkpoints[:-max_checkpoints]:
            logger.info("deleted old checkpoint", path=old)
            old.unlink()


def restore_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: torch.GradScaler,
    device: str,
) -> int:
    state = torch.load(path, map_location=device, weights_only=True)

    model.load_state_dict(state["model"])
    optimizer.load_state_dict(state["optimizer"])
    scheduler.load_state_dict(state["scheduler"])
    scaler.load_state_dict(state["scaler"])

    step = state["step"]
    logger.info("restored checkpoint", path=path, step=step)
    return step
