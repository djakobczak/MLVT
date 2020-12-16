import logging

from flask.views import MethodView
import torch

from mlvt.model.models import Model
from mlvt.config.config_manager import ConfigManager
from mlvt.server.exceptions import ModelException
from mlvt.server.exceptions import ActionOngoingException
from mlvt.server.extensions import executor
from mlvt.server.file_utils import get_current_config


LOG = logging.getLogger('MLVT')


class BaseView(MethodView):
    def __init__(self):
        self.init_cm()

    def init_cm(self):
        self.cm = ConfigManager(get_current_config())

    def _numeric_to_classname(self):
        return dict((v, k) for k, v in self.cm.get_label_mapping().items())


class ActionView(BaseView):
    def run_action(self, action, executable, **kwargs):
        """Add action to execution in background thread

        Args:
            action (Action): enumeration that defines action type
            executable (obj): function that will be executed in separate thread
        """
        self._fail_if_ongoing_action(action)
        executor.submit_stored(action, executable, **kwargs)
        LOG.info(f"New action ({action.value}) added to execution")

    def _fail_if_ongoing_action(self, action):
        """Raise exception if action with specified action type is already ongoing

        Args:
            action (Action): enumeration that defines action type

        Raises:
            ActionOngoingException: specified action is ongoing
        """
        if action in executor.futures._futures and \
                not executor.futures.done(action):
            raise ActionOngoingException(
                f"Wait for action ({action}) to be finished")


class ModelIOView(BaseView):
    def load_training_model(self):
        return self._load_model(self.cm.get_training_model())

    def load_best_model(self):
        return self._load_model(self.cm.get_best_model())

    def _load_model(self, path):
        try:
            return Model(state=Model.load(path),
                         training_model_path=self.cm.get_training_model(),
                         best_model_path=self.cm.get_best_model())
        except FileNotFoundError:
            return Model(training_model_path=self.cm.get_training_model(),
                         best_model_path=self.cm.get_best_model(),
                         overwrite=True)
            raise ModelException('Error while loading trained model')

    def save_training_model(self, model=None, custom_path=None):
        training_model_path = custom_path or self.cm.get_training_model()
        torch.save(model.model_conv, training_model_path)

    def save_best_model(self, model, custom_path=None):
        best_model_path = custom_path or self.cm.get_best_model()
        torch.save(model.model_conv, best_model_path)
