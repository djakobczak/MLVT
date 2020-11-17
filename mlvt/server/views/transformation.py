from mlvt.model.logger import LOG
from mlvt.model.preprocessing.loader import DataLoader
from mlvt.server.views.base import BaseView
from mlvt.server.utils import DatasetType


class TransformView(BaseView):
    def put(self, dataset_type):
        dl = DataLoader(self.cm)
        if dataset_type == 'all':
            for dataset in DatasetType:
                srcs, dsts = self._dataset_type_to_paths(dataset.value)
                dl.copy_multiple_paths(srcs, dsts)
        else:
            srcs, dsts = self._dataset_type_to_paths(dataset_type)
            dl.copy_multiple_paths(srcs, dsts)
        return '', 200

    def _dataset_type_to_paths(self, dataset_type):
        if dataset_type == DatasetType.UNLABELLED.value:
            LOG.debug('Return paths for unlabelled images.')
            return ([self.cm.get_raw_unl_dir()],
                    [self.cm.get_unl_transformed_dir()])
        elif dataset_type == DatasetType.TRAIN.value:
            LOG.debug('Return paths for unlabelled images.')
            return (self.cm.get_raw_train_dirs(),
                    self.cm.get_train_transformed_dirs())
        elif dataset_type == DatasetType.TEST.value:
            LOG.debug('Return paths for test images.')
            return (self.cm.get_raw_test_dirs(),
                    self.cm.get_test_transformed_dirs())
        elif dataset_type == DatasetType.VALIDATION.value:
            LOG.debug('Return paths for validation images.')
            return (self.cm.get_raw_validation_dirs(),
                    self.cm.get_validation_transformed_dirs())
