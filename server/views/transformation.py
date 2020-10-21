from flask import request

from dral.logger import LOG
from dral.preprocessing.loader import DataLoader
from dral.utils import get_before_tensor_transforms
from server.views.base import BaseView
from server.utils import DatasetType


class TransformView(BaseView):
    def put(self):
        dataset_type = request.args.get('dataset_type')
        # load data and perform preprocessing
        dl = DataLoader(self.cm, get_before_tensor_transforms())
        srcs, dsts = self._dataset_type_to_paths(dataset_type)
        dl.copy_multiple_paths(srcs, dsts)

        return 200

    def _dataset_type_to_paths(self, dataset_type):
        print(dataset_type)
        if dataset_type == DatasetType.UNLABELLED.value:
            LOG.debug('Return paths for unlabelled images.')
            return ([self.cm.get_raw_unl_dir()],
                    [self.cm.get_unl_transformed_dir()])
        elif dataset_type == DatasetType.TRAIN.value:
            return None, None   # !TODO
        elif dataset_type == DatasetType.TEST.value:
            LOG.debug('Return paths for test images.')
            return (self.cm.get_raw_test_dirs(),
                    self.cm.get_test_transformed_dirs())
