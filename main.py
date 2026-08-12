from pydantic_settings import BaseSettings, CliApp, CliSubCommand

from src.gsm.eval import EvalArgs as Gsm8kEvalArgs
from src.gsm.grpo import GrpoArgs as GsmGrpoArgs
from src.gsm.sft import SftArgs as Gsm8kSftArgs
from src.infer import InferArgs
from src.pretrain import PreTrainArgs


class CliArgs(BaseSettings):
    pretrain: CliSubCommand[PreTrainArgs]
    infer: CliSubCommand[InferArgs]
    gsm_eval: CliSubCommand[Gsm8kEvalArgs]
    gsm_sft: CliSubCommand[Gsm8kSftArgs]
    gsm_grpo: CliSubCommand[GsmGrpoArgs]

    def cli_cmd(self) -> None:
        CliApp.run_subcommand(self)


def main() -> None:
    CliApp.run(CliArgs)


if __name__ == "__main__":
    main()
