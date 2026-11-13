import logging
import sys


TRACE_LEVEL = 5


def setup_logging(debug: bool = False):
    level = logging.DEBUG if debug else logging.INFO

    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler()],
    )
    logging.addLevelName(TRACE_LEVEL, "TRACE")
    logging.Logger.trace = trace


def trace(self, message, *args, **kwargs):
    if self.isEnabledFor(TRACE_LEVEL):
        self._log(TRACE_LEVEL, message, args, **kwargs)


class RedirectStdoutStderr:
    def __init__(self, target):
        self.target = target

    def __enter__(self):
        self.original_stdout = sys.stdout
        self.original_stderr = sys.stderr
        sys.stdout = self.target
        sys.stderr = self.target

    def __exit__(self, exc_type, exc_value, traceback):
        sys.stdout = self.original_stdout
        sys.stderr = self.original_stderr
