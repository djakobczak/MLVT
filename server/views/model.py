from dral.logger import LOG
from dral.models import Model
from server.views.base import MLView


class ModelView(MLView):
    def search(self):
        model = self.load_model()
        return str(model.model_conv), 200

    def delete(self):
        # overwrite trained model with new untrained
        self.save_model(Model().model_conv)
        LOG.info('Trained model is deleted.')
        return 'Model deleted', 200
