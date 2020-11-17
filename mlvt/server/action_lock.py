import logging   # !TODO split into 2 loggers
from contextlib import ContextDecorator

from mlvt.server.exceptions import ActionLockedException


LOG = logging.getLogger('MLVT')


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
