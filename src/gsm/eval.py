from typing import cast

import tiktoken

from src.gsm.data import Gsm8kSample, load_gsm8k
from src.infer import generate_greedy, generate_top_p
from src.model import Model, ModelLoadArgs


def eval_one_sample(
    model: Model,
    max_len: int,
    tokenizer: tiktoken.Encoding,
    sample: Gsm8kSample,
    device: str,
    use_tqdm: bool = False,
) -> None:
    use_fp16 = device == "cuda"
    greedy_answer = generate_greedy(
        model,
        tokenizer,
        sample.input_ids,
        max_len,
        device,
        use_fp16=use_fp16,
        use_tqdm=use_tqdm,
    )
    top_p_answers = generate_top_p(
        model,
        tokenizer,
        sample.input_ids,
        32,
        max_len,
        temperature=0.7,
        top_p=0.95,
        device=device,
        use_fp16=use_fp16,
        use_tqdm=use_tqdm,
    )
    # TODO(chibo): parse and eval the generated Python code


class EvalArgs(ModelLoadArgs):
    def cli_cmd(self) -> None:
        run_eval(self)


def run_eval(args: EvalArgs) -> None:
    model, tokenizer = args.load_model_and_tokenizer()

    _, test = load_gsm8k(tokenizer, args.model_cfg.max_len)

    for i, sample in enumerate(test):
        if i > 0:
            break
        sample = cast(Gsm8kSample, sample)

        prompt = tokenizer.decode(sample.input_ids.tolist())
        print(prompt)
        print(f"answer: {sample.answer}")
        greedy_answer = generate_greedy(
            model,
            tokenizer,
            prompt,
            args.model_cfg.max_len,
            args.device,
            use_fp16=args.device == "cuda",
            use_tqdm=True,
        )
        print("-" * 120)
        print(greedy_answer[0])
        print(greedy_answer[1])
        answers = generate_top_p(
            model,
            tokenizer,
            sample.input_ids,
            32,
            args.model_cfg.max_len,
            temperature=0.5,
            top_p=0.9,
            device=args.device,
            use_fp16=args.device == "cuda",
            use_tqdm=True,
        )
        for a in answers:
            print("-" * 120)
            print(a[0])
            print(a[1])

        print("=" * 120)
