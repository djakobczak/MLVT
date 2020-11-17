from enum import Enum

from mlvt.server.file_utils import load_json
from mlvt.server.views.base import BaseView


class DataType(Enum):
    TRAIN_RESULTS = 'train_results'


class DataView(BaseView):
    def get(self, datatype):
        if datatype == DataType.TRAIN_RESULTS.value:
            return self._get_train_results()

    def _get_train_results(self):
        path = self.cm.get_train_results_file()
        data = load_json(path)
        return data, 200
