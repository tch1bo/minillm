import datetime

import structlog
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
