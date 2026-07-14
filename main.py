from pydantic_settings import BaseSettings, CliApp, CliSubCommand

from src.pretrain import PreTrainArgs


class CliArgs(BaseSettings):
    pretrain: CliSubCommand[PreTrainArgs]

    def cli_cmd(self) -> None:
        CliApp.run_subcommand(self)


def main() -> None:
    CliApp.run(CliArgs)


if __name__ == "__main__":
    main()
