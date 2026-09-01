"""
This script does GRPO on the gsm8k/train split.
It uses the gsm8k/test split for testing.
"""

import contextlib
import copy
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import NamedTuple, cast

import psutil
import tiktoken
import torch
import tqdm
from pydantic import BaseModel, Field
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.checkpoint import checkpoint
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from src.gsm.data import Gsm8kBatch, load_gsm8k
from src.gsm.eval import (
    GenerationAndExec,
    Gsm8kGeneration,
    generate_for_batch,
)
from src.infer import Generation
from src.model import (
    Config as ModelConfig,
)
from src.model import Model
from src.utils import (
    ListDataset,
    get_logger,
    restore_checkpoint,
    save_checkpoint,
)

logger = get_logger()


class GrpoArgs(BaseModel):
    seed: int = 42
    tokenizer_name: str = "gpt2"
    rollout_batch_size: int = Field(
        default=16,
        description="how many questions from the training dataset to include into one batch for generating rollouts",
    )
    group_size: int = Field(
        default=8, description="how many rollouts to generate for one question"
    )
    mu_steps: int = Field(
        default=3, description="how many gradient updates to do per model update"
    )
    top_p: float = 0.95
    rollout_temp: float = Field(
        default=1.0, description="the temperature for sampling the rollouts"
    )
    eval_temp: float = Field(
        default=0.7, description="the temperature for sampling the rollouts"
    )
    eps: float = Field(default=0.1, description="The epsilon used in ratio clipping")
    beta: float = Field(default=0.004, description="The beta used for KL-divergence")
    num_iterations: int = Field(
        default=10000,
        description="the total number of iterations (one iteration involves `mu_steps` gradient updates",
    )
    min_gradient_update_size: int = Field(
        default=512,
        description="the min number of samples to include per gradient update",
    )
    train_batch_size: int = 4
    eval_batch_size: int = 16
    num_logprob_chunks: int = 4

    max_lr: float = 1e-6
    out_dir: Path = Field(
        default_factory=lambda: Path(
            f"./out/gsm_grpo_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
        )
    )
    save_every: int = 20
    lr_warmup_steps: int = 50
    max_checkpoints: int = Field(
        default=10,
        description="the max number of checkpoints to save. If non-positive, all checkpoints will be saved",
    )
    resume_from_checkpoint: Path | None = None
    model_path: Path | None = None
    model_cfg: ModelConfig = Field(default_factory=ModelConfig)

    def cli_cmd(self) -> None:
        run_gsm_grpo(self)

    @property
    def total_gradient_updates(self) -> int:
        return self.num_iterations * self.mu_steps


class Rollout(NamedTuple):
    g: Generation
    advantage: float


def _rewards(gt_answer: int, gs: list[GenerationAndExec]) -> torch.Tensor:
    # The rewards are:
    #   1.0 if the answer matches the expected answer
    #   -1.0 if there's no integer answer (e.g. the program crashed or was syntactically incorrect)
    #    0.0 otherwise
    rewards = torch.zeros(len(gs), dtype=torch.float32)
    for i, r in enumerate(gs):
        if r.e.answer == gt_answer:
            rewards[i] = 1.0
        elif r.e.answer is None:
            rewards[i] = -1.0
    return rewards


def _advantages(rewards: torch.Tensor) -> torch.Tensor | None:
    if (rewards == rewards[0]).all():
        # All the samples have the same reward
        return None

    # NOTE: Because of the -1/0/1 rewards, we don't divide by the std (that's also what Dr. GRPO does)
    # Doing so would underweigh the groups with {-1, 1} rewards (because they'd have a higher variance)
    return rewards - rewards.mean()


class TokenizedRolloutBatch(NamedTuple):
    input_ids: torch.Tensor
    advantages: torch.Tensor

    # (batch_size, padded_length), a matrix of booleans, True means the token is a padding token
    # the samples in the batch are right-padded
    pad_mask: torch.Tensor

    # (batch_size, padded_length), a matrix of booleans, True means the token should be used in the
    # loss calculation
    loss_mask: torch.Tensor


@dataclass
class RolloutCollator:
    tokenizer: tiktoken.Encoding

    def __call__(self, batch: list[Rollout]) -> TokenizedRolloutBatch:
        # shape is (batch_size, max_len_in_batch)
        shape = (
            len(batch),
            max(len(r.g.input_tokens) + len(r.g.output_tokens) for r in batch),
        )
        input_ids = torch.full(shape, self.tokenizer.eot_token, dtype=torch.long)
        pad_mask = torch.full(shape, True, dtype=torch.bool)
        loss_mask = torch.full(shape, False, dtype=torch.bool)
        for i, r in enumerate(batch):
            in_len, out_len = len(r.g.input_tokens), len(r.g.output_tokens)

            input_ids[i, :in_len] = r.g.input_tokens
            input_ids[i, in_len : in_len + out_len] = r.g.output_tokens
            pad_mask[i, : in_len + out_len] = False
            loss_mask[i, in_len : in_len + out_len] = True
        advantages = torch.tensor([r.advantage for r in batch], dtype=torch.float32)
        return TokenizedRolloutBatch(input_ids, advantages, pad_mask, loss_mask)


def copy_and_freeze_model(model: Model) -> Model:
    new_model = copy.deepcopy(model)
    new_model.eval()
    for p in new_model.parameters():
        p.requires_grad_(False)
    return new_model


def chunked_log_probs(
    model: Model,
    input_ids: torch.Tensor,
    pad_mask: torch.Tensor,
    *,
    num_chunks: int,
    is_train: bool,
    use_fp16: bool,
) -> torch.Tensor:
    # NOTE: see `chunked_cross_entropy_loss` for more details on why chunking is needed

    with (
        torch.autocast(input_ids.device.type, dtype=torch.float16, enabled=use_fp16),
        contextlib.nullcontext() if is_train else torch.no_grad(),
    ):
        hidden = model.forward_no_lm_head(
            input_ids,
            torch.arange(input_ids.shape[1], device=input_ids.device),
            pad_mask,
        )

        def chunk_fn(
            hidden_chunk: torch.Tensor, targets_chunk: torch.Tensor
        ) -> torch.Tensor:
            logits = model.lm_head(hidden_chunk)
            probs = logits.log_softmax(dim=-1)[:, :-1, :]
            return torch.gather(probs, -1, targets_chunk).squeeze(-1)

        probs_chunks: list[torch.Tensor] = []
        targets = input_ids[:, 1:].unsqueeze(-1)
        for h, t in zip(
            hidden.chunk(num_chunks, dim=0),
            targets.chunk(num_chunks, dim=0),
            strict=True,
        ):
            if is_train:
                t = checkpoint(chunk_fn, h, t, use_reentrant=False)
            else:
                t = chunk_fn(h, t)
            probs_chunks.append(t)

        return torch.cat(probs_chunks)


def start_system_metrics_worker(writer: SummaryWriter):
    interval = 15.0

    def worker() -> None:
        proc = psutil.Process()
        t0 = time.time()
        while True:
            elapsed = int(time.time() - t0)
            writer.add_scalar(
                "system/ram_used_gb", psutil.virtual_memory().used / 2**30, elapsed
            )
            writer.add_scalar(
                "system/proc_rss_gb", proc.memory_info().rss / 2**30, elapsed
            )
            time.sleep(interval)

    threading.Thread(target=worker, daemon=True).start()
    logger.info("started the system metrics logger", interval=interval)


def run_gsm_grpo(args: GrpoArgs) -> None:
    torch.manual_seed(args.seed)
    args.out_dir.mkdir(exist_ok=True, parents=True)
    (args.out_dir / "args.json").write_text(args.model_dump_json(indent=2))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_fp16 = device == "cuda"

    # Load the tokenizer
    tokenizer = tiktoken.get_encoding(args.tokenizer_name)

    # Load the data
    train, test = load_gsm8k(
        tokenizer,
        train_batch_size=args.rollout_batch_size,
        test_batch_size=args.eval_batch_size,
        max_len=args.model_cfg.max_len,
    )

    # Load the model
    model = Model(args.model_cfg, vocab_size=tokenizer.max_token_value + 1).to(device)

    # Init the optimizer and the LR scheduler
    optimizer = torch.optim.Adam(params=model.parameters(), lr=args.max_lr, fused=True)

    def lr_lambda(step: int):
        return min(1.0, (step + 1) / args.lr_warmup_steps)

    lr_scheduler = LambdaLR(optimizer, lr_lambda)
    scaler = torch.amp.GradScaler("cuda", enabled=use_fp16)

    def global_mu_step(iteration: int, mu_step: int) -> int:
        assert mu_step < args.mu_steps
        return iteration * args.mu_steps + mu_step

    # Load the model
    if args.resume_from_checkpoint is None:
        assert args.model_path is not None
        model.load_state_dict(torch.load(args.model_path))
        start_iteration = 0
    else:
        start_iteration = (
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

    ref_model = copy_and_freeze_model(model)
    ref_model.eval()

    train_iterator, epoch = iter(train), 0

    def next_prompt_batch() -> Gsm8kBatch:
        nonlocal train_iterator, epoch
        try:
            batch = next(train_iterator)
        except StopIteration:
            logger.info("finished epoch", num_epoch=epoch)
            epoch += 1
            train_iterator = iter(train)
            batch = next(train_iterator)

        return cast(Gsm8kBatch, batch)

    logger.info(
        "starting training",
        min_gradient_update_size=args.min_gradient_update_size,
        total_gradient_updates=args.total_gradient_updates,
        group_size=args.group_size,
        start_iteration=start_iteration,
    )
    writer = SummaryWriter(log_dir=args.out_dir)
    start_system_metrics_worker(writer)

    running_kl_div = torch.zeros((), device=device)
    running_token_count = torch.zeros((), dtype=torch.long, device=device)
    fraction_of_clipped_ratios = torch.zeros((), device=device)
    for iteration in tqdm.trange(
        start_iteration, start_iteration + args.num_iterations, desc="training"
    ):
        step = global_mu_step(iteration, mu_step=0)

        # Step 1 - generate the rollouts and compute the advantages
        rollouts: list[Rollout] = []
        num_none_groups, total_num_groups = 0, 0
        total_generation_time = 0.0
        all_rewards: list[torch.Tensor] = []
        total_num_out_tokens = 0
        num_truncated_gens = 0
        model.eval()
        while len(rollouts) < args.min_gradient_update_size:
            # Sample a batch of prompts
            prompt_batch = next_prompt_batch()
            total_num_groups += len(prompt_batch)

            # Generate the rollouts
            start_time = time.perf_counter()
            batched_rollouts = generate_for_batch(
                model,
                tokenizer,
                prompt_batch,
                device,
                max_len=args.model_cfg.max_len,
                eval_greedy=False,
                eval_top_p=args.group_size,
                top_p=args.top_p,
                temp=args.rollout_temp,
                tqdm_kwargs=dict(desc="top_p", leave=False),
            )
            total_generation_time += time.perf_counter() - start_time

            for br in batched_rollouts:
                total_num_out_tokens += sum(len(x.g.output_tokens) for x in br.top_p)
                non_truncated = [x for x in br.top_p if not x.g.truncated]
                num_truncated_gens += len(br.top_p) - len(non_truncated)
                if not non_truncated:
                    continue

                all_rewards.append(_rewards(br.answer, non_truncated))
                if (advantages := _advantages(all_rewards[-1])) is None:
                    num_none_groups += 1
                    continue
                rollouts.extend(
                    (
                        Rollout(p.g, advantage.item())
                        for p, advantage in zip(non_truncated, advantages, strict=True)
                    )
                )

        rewards = torch.cat(all_rewards)
        mean_reward = rewards.mean().item()
        total_rollouts = total_num_groups * args.group_size
        ok_answer = (rewards == 1).int().sum().item() / total_rollouts
        wrong_answer = (rewards == 0).int().sum().item() / total_rollouts
        broken_program = (rewards == -1).int().sum().item() / total_rollouts
        mean_output_tokens = total_num_out_tokens / total_rollouts
        ratio_truncated = num_truncated_gens / total_rollouts
        logger.info(
            "finished generation",
            gen_time=f"{total_generation_time:.3f}s",
            ok_rollouts=len(rollouts),
            total_rollouts=total_rollouts,
            mean_out_tokens=mean_output_tokens,
            mean_reward=rewards.mean().item(),
            ok_answer=ok_answer,
            wrong_answer=wrong_answer,
            broken_code=broken_program,
            ratio_truncated=ratio_truncated,
        )

        writer.add_scalar("gen/num_ok_rollouts", len(rollouts), step)
        writer.add_scalar("gen/total_rollouts", total_rollouts, step)
        writer.add_scalar("gen/generation_time", total_generation_time, step)
        writer.add_scalar("gen/mean_output_tokens", mean_output_tokens, step)
        writer.add_scalar("gen/mean_reward", mean_reward, step)
        writer.add_scalar("gen/ratio_truncated", ratio_truncated, step)
        writer.add_scalar("gen/ok_answer", ok_answer, step)
        writer.add_scalar("gen/wrong_answer", wrong_answer, step)
        writer.add_scalar("gen/broken_program", broken_program, step)

        # Step 2 - compute the GRPO loss and do the gradient updates
        # These contain one tensor per mini-batch
        old_logprobs: list[torch.Tensor] = []
        ref_logprobs: list[torch.Tensor] = []
        tr_batches: list[TokenizedRolloutBatch] = list(
            DataLoader(
                ListDataset(rollouts),
                collate_fn=RolloutCollator(tokenizer),
                shuffle=True,
                batch_size=args.train_batch_size,
            )
        )
        for mu_step in range(0, args.mu_steps):
            running_kl_div.zero_()
            running_token_count.zero_()
            fraction_of_clipped_ratios.zero_()
            for i, tr_batch in enumerate(
                tqdm.tqdm(tr_batches, leave=False, desc=f"mu_step {mu_step}")
            ):
                tr_batch = cast(TokenizedRolloutBatch, tr_batch)
                input_ids, pad_mask, adv = (
                    tr_batch.input_ids.to(device),
                    tr_batch.pad_mask.to(device),
                    tr_batch.advantages.to(device),
                )

                # Compute the probabilities of the reference, old and the current models
                logprobs = chunked_log_probs(
                    model,
                    input_ids,
                    pad_mask,
                    num_chunks=args.num_logprob_chunks,
                    is_train=True,
                    use_fp16=use_fp16,
                ).float()
                if mu_step == 0:
                    # NOTE: with the current model implementation, there's no difference between
                    # model.train() and model.eval() (we don't use dropout/batch norm/etc.)
                    # Hence it's safe to simply do logprobs.detach() here
                    old_logprobs.append(logprobs.detach())
                    ref_logprobs.append(
                        chunked_log_probs(
                            ref_model,
                            input_ids,
                            pad_mask,
                            num_chunks=1,
                            is_train=False,
                            use_fp16=use_fp16,
                        ).float()
                    )

                # Compute the GRPO-objective
                loss_mask = tr_batch.loss_mask.to(device)[:, 1:]
                ratio = torch.exp(logprobs - old_logprobs[i])
                clipped = ratio.clamp(1 - args.eps, 1 + args.eps)
                obj = torch.min(ratio * adv[:, None], clipped * adv[:, None])

                fraction_of_clipped_ratios += (
                    ((ratio != clipped) & loss_mask).float().sum().div_(loss_mask.sum())
                )

                # Approximate the KL-divergence
                r = ref_logprobs[i] - logprobs
                kl_div = r.exp() - r - 1
                running_kl_div += (kl_div * loss_mask).sum().detach()
                running_token_count += loss_mask.sum().detach()

                # Get the final loss
                loss = (
                    ((kl_div * args.beta - obj) * loss_mask).sum()
                    / args.model_cfg.max_len
                    / len(rollouts)
                )
                scaler.scale(loss).backward()

            # End of the mu_step
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scale_before = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            if scaler.get_scale() >= scale_before:
                lr_scheduler.step()
            optimizer.zero_grad()

            writer.add_scalar(
                "training/grad_norm",
                grad_norm.item(),
                global_mu_step(iteration, mu_step),
            )
            writer.add_scalar(
                "training/num_tokens",
                running_token_count.item(),
                global_mu_step(iteration, mu_step),
            )
            writer.add_scalar(
                "training/kl_div",
                (running_kl_div / running_token_count).item(),
                global_mu_step(iteration, mu_step),
            )
            writer.add_scalar(
                "training/fraction_of_clipped_ratios",
                fraction_of_clipped_ratios.div_(len(rollouts)).item(),
                global_mu_step(iteration, mu_step),
            )

        # Step 3 - periodically run evals and save weights
        if iteration % args.save_every == 0:
            save_checkpoint(
                args.out_dir,
                model,
                optimizer,
                lr_scheduler,
                scaler,
                step=iteration,
                max_checkpoints=args.max_checkpoints,
            )

            test_gen: list[Gsm8kGeneration] = []
            for test_batch in tqdm.tqdm(test, desc="running validation"):
                test_batch = cast(Gsm8kBatch, test_batch)
                test_gen.extend(
                    generate_for_batch(
                        model,
                        tokenizer,
                        test_batch,
                        device,
                        max_len=args.model_cfg.max_len,
                        eval_greedy=True,
                        eval_top_p=args.group_size,
                        top_p=args.top_p,
                        temp=args.eval_temp,
                    )
                )
            pass_1 = len(
                [g for g in test_gen if g.greedy and g.answer == g.greedy.e.answer]
            ) / len(test_gen)

            pass_k = len(
                [g for g in test_gen if any(p.e.answer == g.answer for p in g.top_p)]
            ) / len(test_gen)

            logger.info("finished the validation step", pass_1=pass_1, pass_k=pass_k)

            writer.add_scalar("test/pass_1", pass_1, step)
            writer.add_scalar("test/pass_k", pass_k, step)

            for i, sample in enumerate(test_gen[:10]):
                if sample.greedy is not None:
                    writer.add_text(
                        f"test/greedy_generation_{i}",
                        sample.greedy.to_tensorboard_md(sample.answer),
                        step,
                    )
                if sample.top_p:
                    writer.add_text(
                        f"test/top_p_1_generation_{i}",
                        sample.top_p[0].to_tensorboard_md(sample.answer),
                        step,
                    )

            greedy_truncated = [
                g for g in test_gen if g.greedy is not None and g.greedy.g.truncated
            ]
            writer.add_scalar(
                "test/ratio_truncated_greedy",
                len(greedy_truncated) / len(test_gen),
                step,
            )
            for i, sample in enumerate(greedy_truncated):
                assert sample.greedy is not None
                writer.add_text(
                    f"test/greedy_generation_{i}",
                    sample.greedy.to_tensorboard_md(sample.answer),
                    step,
                )
    writer.close()
