import os
import subprocess
import tempfile
from pathlib import Path
from typing import cast

import tiktoken
import tqdm
from pydantic import BaseModel, RootModel

from src.gsm.data import Gsm8kSample, load_gsm8k
from src.infer import generate_greedy, generate_top_p
from src.model import Model, ModelLoadArgs
from src.utils import get_logger

logger = get_logger()


class ExecutionResult(BaseModel):
    returncode: int | None = None
    exception_str: str | None = None
    stdout: str | None = None
    answer: int | None = None

    def is_correct(self, expected_answer: int) -> bool:
        return self.answer is not None and self.answer == expected_answer


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
    except ValueError as e:
        return ExecutionResult(exception_str=str(e), stdout=result.stdout)

    return ExecutionResult(answer=answer)


class GenerationResult(BaseModel):
    code: str
    prob: float
    exec_result: ExecutionResult

    def is_correct(self, expected_answer: int) -> bool:
        return self.exec_result.is_correct(expected_answer)


class SampleResult(BaseModel):
    greedy: GenerationResult
    top_p: list[GenerationResult]
    expected_answer: int

    def greedy_correct(self) -> bool:
        return self.greedy.is_correct(self.expected_answer)

    def any_top_p_correct(self) -> bool:
        return any(r.is_correct(self.expected_answer) for r in self.top_p)

    def ratio_top_p_correct(self) -> float:
        return sum(int(r.is_correct(self.expected_answer)) for r in self.top_p) / len(
            self.top_p
        )


def eval_one_sample(
    model: Model,
    max_len: int,
    tokenizer: tiktoken.Encoding,
    sample: Gsm8kSample,
    device: str,
    use_tqdm: bool = False,
) -> SampleResult:
    use_fp16 = device == "cuda"
    greedy_code, greed_prob = generate_greedy(
        model,
        tokenizer,
        sample.input_ids,
        max_len,
        device,
        use_fp16=use_fp16,
        tqdm_desc="greedy sampling" if use_tqdm else None,
    )
    greedy_eval = _run_one_sample(greedy_code)

    top_p_codes = generate_top_p(
        model,
        tokenizer,
        sample.input_ids,
        32,
        max_len,
        temperature=0.7,
        top_p=0.95,
        device=device,
        use_fp16=use_fp16,
        tqdm_desc="top-p sampling" if use_tqdm else None,
    )
    top_p_codes = sorted(top_p_codes, key=lambda x: x[1], reverse=True)
    top_p_evals = [_run_one_sample(c[0]) for c in top_p_codes]

    return SampleResult(
        greedy=GenerationResult(
            code=greedy_code, prob=greed_prob, exec_result=greedy_eval
        ),
        top_p=[
            GenerationResult(code=c, prob=p, exec_result=er)
            for ((c, p), er) in zip(top_p_codes, top_p_evals, strict=True)
        ],
        expected_answer=sample.answer,
    )


class EvalArgs(ModelLoadArgs):
    num_samples: int | None = None
    out_dir: Path = Path("/tmp/gsm_eval")

    def cli_cmd(self) -> None:
        run_eval(self)


def run_eval(args: EvalArgs) -> None:
    model, tokenizer = args.load_model_and_tokenizer()

    _, test = load_gsm8k(tokenizer, args.model_cfg.max_len)

    num_samples = args.num_samples if args.num_samples is not None else len(test)
    logger.info(
        "evaluating on the gsm8k test split",
        num_samples=num_samples,
        out_dir=args.out_dir,
    )
    results: list[SampleResult] = []
    for i, sample in zip(tqdm.trange(num_samples), test):
        sample = cast(Gsm8kSample, sample)
        result = eval_one_sample(
            model,
            args.model_cfg.max_len,
            tokenizer,
            sample,
            args.device,
            use_tqdm=True,
        )
        out_path = args.out_dir / f"{i}.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(result.model_dump_json(indent=2))
        results.append(result)

    all_out_path = args.out_dir / "all.json"
    all_out_path.write_text(
        RootModel[list[SampleResult]](results).model_dump_json(indent=2)
    )
    logger.info("wrote all results", out_path=all_out_path)

    greedy_pass_1 = sum(int(r.greedy_correct()) for r in results) / len(results)
    pass_32 = sum(int(r.any_top_p_correct()) for r in results) / len(results)
    pass_32_ratio = sum(r.ratio_top_p_correct() for r in results) / len(results)
    logger.info("done evaluating")
    logger.info("  greedy pass_1:", value=greedy_pass_1)
    logger.info("  top_p pass_32:", value=pass_32)
    logger.info("  top_p pass_32 ratio:", value=pass_32_ratio)
