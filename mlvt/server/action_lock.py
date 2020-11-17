from contextlib import ContextDecorator

from mlvt.model.logger import LOG   # !TODO split into 2 loggers
from mlvt.server.exceptions import ActionLockedException


class lock(ContextDecorator):
    locked = False

    def __enter__(self):
        if self.locked:
            raise ActionLockedException(
                "There is ongoing action, please wait until is finished")
        self.locked = True
        LOG.info("Server locked")
        return self

    def __exit__(self, type, value, traceback):
        self.locked = False
        LOG.info("Server unlocked")
