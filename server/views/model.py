from connexion import request

from dral.logger import LOG
from dral.models import Model
from server.views.base import MLView
from server.file_utils import (fail_if_headpath_not_exist,
                               create_subdirs_if_not_exist,
                               save_json)


class ModelView(MLView):
    def search(self):
        model = self.load_model()
        return str(model.model_conv), 200

    def delete(self):
        # overwrite trained model with new untrained
        self.save_model(Model())
        save_json(self.cm.get_predictions_file(), {})
        LOG.info('Model is resetted and predictions is pruned')
        return 'Model deleted', 200

    def put(self, force):
        path = request.get_data(as_text=True)
        if force:
            create_subdirs_if_not_exist(path)
        fail_if_headpath_not_exist(path)
        self.save_model(path=path)
        return 'OK', 200
