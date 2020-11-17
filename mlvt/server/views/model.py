from connexion import request

from mlvt.model.logger import LOG
from mlvt.model.models import Model
from mlvt.server.views.base import MLView
from mlvt.server.file_utils import (fail_if_headpath_not_exist,
                                    create_subdirs_if_not_exist,
                                    save_json, purge_json_file)
from mlvt.server.views.annotation import AnnotationsView
from mlvt.server.views.train import EMPTY_TRAIN_RESULTS


class ModelView(MLView):
    def search(self):
        model = self.load_model()
        return str(model.model_conv), 200

    def delete(self, clear_annotations, clear_history):
        # overwrite trained model with new untrained and clear all history data
        self.save_model(Model())
        save_json(self.cm.get_predictions_file(), {})
        purge_json_file(self.cm.get_last_user_test_path())
        if clear_history:
            purge_json_file(self.cm.get_test_results_file())
            purge_json_file(
                self.cm.get_train_results_file(),
                EMPTY_TRAIN_RESULTS)
        if clear_annotations:
            av = AnnotationsView()
            av.put(True, 'all', True)

        LOG.info('Model is resetted and predictions is pruned')
        return 'Model deleted', 200

    def put(self, force):
        path = request.get_data(as_text=True)
        if force:
            create_subdirs_if_not_exist(path)
        fail_if_headpath_not_exist(path)
        self.save_model(path=path)
        return '', 200
