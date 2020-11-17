import os

import torch
from torch.utils.data import DataLoader

from mlvt.model.models import Model
from mlvt.config.config_manager import ConfigManager
from mlvt.model.datasets import UnlabelledDataset, LabelledDataset
from mlvt.model.utils import get_resnet_test_transforms, \
    get_resnet_train_transforms

from mlvt.server.exceptions import AnnotationException, ModelException
from mlvt.server.file_utils import is_json_empty
from mlvt.server.config import CONFIG_NAME


class BaseAction:
    def __init__(self):
        self.cm = ConfigManager(CONFIG_NAME)


class MLAction(BaseAction):
    def __init__(self):
        super().__init__()
        self.unl_dataset = None
        self.train_dataset = None
        self.test_dataset = None
        self.validation_loader = None
        self.unl_loader = None
        self.train_loader = None
        self.test_loader = None

    # csv with annotations can not be empty
    def get_unl_loader(self):
        annotation_path = self.cm.get_unl_annotations_path()
        self._fail_if_file_is_empty(annotation_path)
        self.unl_dataset = UnlabelledDataset(
            annotation_path,
            get_resnet_test_transforms())

        self.unl_loader = DataLoader(
            self.unl_dataset, batch_size=self.cm.get_batch_size(),
            shuffle=True, num_workers=0)
        return self.unl_loader

    def get_train_loader(self, batch_size):
        annotation_path = self.cm.get_train_annotations_path()
        self._fail_if_file_is_empty(annotation_path)
        self.train_dataset = LabelledDataset(
            annotation_path,
            get_resnet_train_transforms())

        self.train_loader = DataLoader(
            self.train_dataset, batch_size=batch_size,
            shuffle=True, num_workers=0)
        return self.train_loader

    def get_test_loader(self):
        annotation_path = self.cm.get_test_annotations_path()
        self._fail_if_file_is_empty(annotation_path)
        if not self.test_dataset:
            self.test_dataset = LabelledDataset(
                annotation_path,
                get_resnet_test_transforms())

            self.test_loader = DataLoader(
                self.test_dataset, batch_size=self.cm.get_batch_size(),
                shuffle=True, num_workers=0)
        return self.test_loader

    def get_validation_loader(self, batch_size):
        annotation_path = self.cm.get_validation_annotations_path()
        self._fail_if_file_is_empty(annotation_path)
        if not self.validation_loader:
            self.validation_dataset = LabelledDataset(
                annotation_path,
                get_resnet_test_transforms())

            self.validation_loader = DataLoader(
                self.validation_dataset, batch_size=batch_size,
                shuffle=True, num_workers=0)
        return self.validation_loader

    def _fail_if_file_is_empty(self, path):
        if not os.path.isfile(path) or not is_json_empty(path):
            raise AnnotationException(
                f'Annotation file ({path}) does not exist or is empty')

    def load_model(self):
        try:
            model = Model.load(self.cm.get_model_trained())
        except FileNotFoundError:
            raise ModelException('Error while loading trained model')
        return Model(model)

    def save_model(self, model=None, path=None):
        model = model if model else self.load_model()
        path = path if path else self.cm.get_model_trained()
        torch.save(model.model_conv, path)
