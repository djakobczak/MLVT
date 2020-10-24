from flask import jsonify

from dral.logger import LOG
from dral.models import Model
from server.views.base import MLView
from server.action_lock import lock


# !TODO add context manager for load/save flow
class TrainView(MLView):
    @lock()
    def search(self, batch_size=None, query=None):
        model = self.load_model()
        losses, accs = model.train(
            self.get_train_loader(),
            self.cm.get_epochs())
        LOG.info(f'losses: {losses}, accs: {accs}')
        self.save_model(model)
        return jsonify(losses=losses, accuracy=accs)
