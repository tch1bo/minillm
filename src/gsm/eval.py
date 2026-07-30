import os
import subprocess
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Sequence, cast

import tiktoken
import torch
import tqdm
from pydantic import BaseModel, Field, RootModel

from src.gsm.data import Gsm8kBatch, load_gsm8k
from src.infer import Generation, generate_greedy, generate_top_p
from src.model import Model, ModelLoadArgs
from src.utils import get_logger

logger = get_logger()


class ExecutionResult(BaseModel):
    returncode: int | None = None
    exception_str: str | None = None
    stdout: str | None = None
    answer: int | None = None


def _run_one_sample(code: str, timeout: float = 1.0) -> ExecutionResult:
    code += "\n\nprint(simple_math_problem(), end='', flush=True)"

    with tempfile.TemporaryDirectory() as temp_dir:
        path = os.path.join(temp_dir, "sol.py")
        with open(path, "w") as f:
            f.write(code)
        try:
            cmd = ["python3", "-I", path]
            wrapped = [
                "bash",
                "-c",
                'ulimit -v 262144 -t 5 -u 32 -f 1024; exec "$@"',
                "--",
                *cmd,
            ]
            result = subprocess.run(
                wrapped,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=temp_dir,
                env={},
                start_new_session=True,
            )
        except subprocess.TimeoutExpired:
            return ExecutionResult(exception_str="timeout")

    if result.returncode != 0:
        return ExecutionResult(returncode=result.returncode)

    try:
        # The code samples in TinyGSM use / instead of //, so the model learns to use it as well.
        # To handle this, we first convert the result to float and then to an int.
        answer = int(float(result.stdout))
    except (ValueError, OverflowError) as e:
        return ExecutionResult(exception_str=str(e), stdout=result.stdout)

    return ExecutionResult(answer=answer)


class GenerationAndExec(BaseModel):
    g: Generation
    e: ExecutionResult


class Gsm8kGeneration(BaseModel):
    # The ground truth answer
    answer: int

    # The solutions generated using greedy sampling (size: batch_size)
    # (if greedy sampling was not requested, is set to None)
    greedy: GenerationAndExec | None

    # The solutions generated using top_p sampling (size: (batch_size, num_samples))
    # (if top_p sampling was not requested, the list will be empty)
    top_p: list[GenerationAndExec]


def generate_for_batch(
    model: Model,
    args: EvalArgs,
    tokenizer: tiktoken.Encoding,
    batch: Gsm8kBatch,
    device: str,
    use_tqdm: bool = False,
) -> list[Gsm8kGeneration]:
    use_fp16 = device == "cuda"

    greedy: Sequence[GenerationAndExec | None] = [None] * len(batch)
    if args.eval_greedy:
        gs = generate_greedy(
            model,
            tokenizer,
            batch.input_ids,
            batch.pad_mask,
            args.model_cfg.max_len,
            device,
            use_fp16=use_fp16,
            tqdm_desc="greedy sampling" if use_tqdm else None,
        )
        es = [_run_one_sample(g.output) for g in gs]
        greedy = [GenerationAndExec(g=g, e=e) for (g, e) in zip(gs, es, strict=True)]

    top_p: list[list[GenerationAndExec]] = [[]] * len(batch)
    if args.eval_top_p is not None:
        gss = generate_top_p(
            model,
            tokenizer,
            batch.input_ids,
            batch.pad_mask,
            args.eval_top_p,
            args.model_cfg.max_len,
            temperature=args.temp,
            top_p=args.top_p,
            device=device,
            use_fp16=use_fp16,
            tqdm_desc="top-p sampling" if use_tqdm else None,
        )
        ess = [[_run_one_sample(g.output) for g in gs] for gs in gss]
        top_p = [
            [GenerationAndExec(g=g, e=e) for (g, e) in zip(gs, es, strict=True)]
            for (gs, es) in zip(gss, ess, strict=True)
        ]

    return [
        Gsm8kGeneration(answer=a, greedy=g, top_p=p)
        for (a, g, p) in zip(batch.answers.tolist(), greedy, top_p, strict=True)
    ]


class EvalArgs(ModelLoadArgs):
    num_batches: int | None = None
    out_path: Path = Field(
        default_factory=lambda: Path(
            f"/tmp/gsm_eval_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.json"
        )
    )
    seed: int = 42
    batch_size: int = 16
    eval_greedy: bool = True
    eval_top_p: int | None = Field(
        default=None,
        description="if None, no top_p samples will be generated. If not-None, will generate that many top_p samples",
    )
    top_p: float = 0.95
    temp: float = 0.7

    def cli_cmd(self) -> None:
        run_eval(self)


def print_stats(gs: list[Gsm8kGeneration], args: EvalArgs) -> None:
    logger.info("eval results")
    if args.eval_greedy:
        ok = [g for g in gs if g.greedy and g.answer == g.greedy.e.answer]
        logger.info("  greedy pass_1:", value=len(ok) / len(gs))

    if args.eval_top_p is not None:
        logger.info(
            "  top_p", num_samples=args.eval_top_p, temp=args.temp, top_p=args.top_p
        )
        pass_1 = [g for g in gs if g.top_p and g.answer == g.top_p[0].e.answer]
        logger.info("    top_p pass_1:", value=len(pass_1) / len(gs))

        pass_n = [g for g in gs if any(g.answer == p.e.answer for p in g.top_p)]
        logger.info("    top_p pass_n:", value=len(pass_n) / len(gs))

        total_top_p_samples = sum([len(g.top_p) for g in gs])
        num_pass = sum(sum(g.answer == p.e.answer for p in g.top_p) for g in gs)
        logger.info("    top_p pass_rate:", value=num_pass / total_top_p_samples)


def run_eval(args: EvalArgs) -> None:
    torch.manual_seed(args.seed)
    model, tokenizer = args.load_model_and_tokenizer()

    _, test = load_gsm8k(
        tokenizer,
        train_batch_size=1,
        test_batch_size=args.batch_size,
        max_len=args.model_cfg.max_len,
    )

    num_batches = args.num_batches if args.num_batches is not None else len(test)
    logger.info(
        "evaluating on the gsm8k test split",
        num_batches=num_batches,
        out_path=args.out_path,
    )
    generations: list[Gsm8kGeneration] = []
    for i, batch in zip(tqdm.trange(num_batches, desc="generating the answers"), test):
        generations.extend(
            generate_for_batch(
                model,
                args,
                tokenizer,
                cast(Gsm8kBatch, batch),
                args.device,
                use_tqdm=False,
            )
        )

    args.out_path.write_text(
        RootModel[list[Gsm8kGeneration]](generations).model_dump_json(indent=2)
    )
    logger.info("wrote all results", out_path=args.out_path)

    print_stats(generations, args)
