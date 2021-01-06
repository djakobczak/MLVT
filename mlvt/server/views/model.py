import logging

from mlvt.model.models import Model
from mlvt.server.views.base import ModelIOView
from mlvt.server.file_utils import save_json, purge_json_file
from mlvt.server.views.annotation import AnnotationsView
from mlvt.server.views.train import EMPTY_TRAIN_RESULTS


LOG = logging.getLogger('MLVT')


class ModelView(ModelIOView):
    def search(self):
        self.init_cm()
        model = self.load_training_model()
        return str(model.model_conv), 200

    def delete(self, clear_annotations, clear_history):
        self.init_cm()
        # overwrite trained model with new untrained and clear all history data
        Model(training_model_path=self.cm.get_training_model(),
              best_model_path=self.cm.get_best_model(),
              model_name=self.cm.get_model_name(), overwrite=True)

        save_json(self.cm.get_predictions_file(), {})
        purge_json_file(self.cm.get_last_user_test_path())
        if clear_history:
            self._clear_history()

        if clear_annotations:
            self._clear_annotations()

        LOG.info('Model is resetted and predictions is pruned')
        return 'Model deleted', 200

    def put(self, clear_annotations, clear_history):
        self.init_cm()
        model = self.load_best_model()
        model.restore_best()
        if clear_history:
            self._clear_history()

        if clear_annotations:
            self._clear_annotations()
        return '', 200

    def _clear_history(self):
        purge_json_file(self.cm.get_test_results_file())
        purge_json_file(
            self.cm.get_train_results_file(),
            EMPTY_TRAIN_RESULTS)

    def _clear_annotations(self):
        av = AnnotationsView()
        av.put(True, 'all', True)
