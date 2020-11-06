import os

from flask.views import MethodView
import torch
from torch.utils.data import DataLoader

from dral.models import Model
from dral.config.config_manager import ConfigManager
from dral.datasets import UnlabelledDataset, LabelledDataset
from dral.logger import LOG
from dral.utils import get_resnet_test_transforms, \
    get_resnet_train_transforms
from server.exceptions import AnnotationException, ModelException
from server.file_utils import is_json_empty, create_subdirs_if_not_exist
from server.exceptions import ActionLockedException
from server.extensions import executor
from server.config import CONFIG_NAME


class BaseView(MethodView):
    def __init__(self):
        self.cm = ConfigManager(CONFIG_NAME)


class ActionView(BaseView):
    def run_action(self, action, executable, **kwargs):
        self._fail_if_ongoing_action(action)
        executor.submit_stored(action, executable, **kwargs)
        LOG.info(f"New action ({action.value}) added to execution")

    def _fail_if_ongoing_action(self, action):
        if action in executor.futures._futures:
            if not executor.futures.done(action):
                raise ActionLockedException("Ongoing action!")
            executor.futures.pop(action)  # !TODO check it


class MLView(BaseView):
    def __init__(self):
        super().__init__()
        self.unl_dataset = None
        self.train_dataset = None
        self.test_dataset = None
        self.unl_loader = None
        self.train_loader = None
        self.test_loader = None

    # csv with annotations can not be empty
    def get_unl_loader(self):
        # if not self.unl_loader:
        annotation_path = self.cm.get_unl_annotations_path()
        self._fail_if_file_is_empty(annotation_path)
        self.unl_dataset = UnlabelledDataset(
            annotation_path,
            get_resnet_test_transforms())

        self.unl_loader = DataLoader(
            self.unl_dataset, batch_size=self.cm.get_batch_size(),
            shuffle=True, num_workers=0)
        return self.unl_loader

    def get_train_loader(self):
        annotation_path = self.cm.get_train_annotations_path()
        self._fail_if_file_is_empty(annotation_path)
        # if not self.train_dataset:
        self.train_dataset = LabelledDataset(
            annotation_path,
            get_resnet_train_transforms())

        self.train_loader = DataLoader(
            self.train_dataset, batch_size=self.cm.get_batch_size(),
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

    def _fail_if_file_is_empty(self, path):  # !TODO could be static or moved somewhere
        if not os.path.isfile(path) or not is_json_empty(path):
            raise AnnotationException(
                'Annotation file does not exist or is empty')

    def load_model(self):
        try:
            model = Model.load(self.cm.get_model_trained())
        except FileNotFoundError:
            raise ModelException('Error while loading trained model')
        return Model(model)

    def save_model(self, model=None, path=None):
        model = model if model else self.load_model()
        path = path if path else self.cm.get_model_trained()
        create_subdirs_if_not_exist(path)
        torch.save(model.model_conv, path)


class ModelIO:  # !TODO to consideration
    def load_model(self, path):
        try:
            model = Model.load(path)
        except FileNotFoundError:
            raise ModelException('Error while loading trained model')
        return Model(model)

    def save_model(self, model=None, path=None):
        model = model if model else self.load_model()
        path = path if path else self.cm.get_model_trained()
        torch.save(model.model_conv, path)
