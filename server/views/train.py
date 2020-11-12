from flask import flash, render_template

from dral.logger import LOG
from server.actions.handlers import train
from server.actions.main import Action
from server.views.base import ActionView
from server.file_utils import load_json, purge_json_file


EMPTY_TRAIN_RESULTS = {
    'train_acc': [], 'train_loss': [],
    'val_acc': [], 'val_loss': [],
    'n_images': []
}


class TrainView(ActionView):

    def search(self, nepochs, reverse):  # !TODO refactoring unclear flow
        train_results = load_json(self.cm.get_train_results_file())

        if not train_results:
            flash('Training history is empty, '
                  'you have to train your model firstly', 'danger')
            return render_template(
                'train.html.j2', results=dict())
        results = self._get_last_n_results(train_results, nepochs, reverse)

        return render_template(
            'train.html.j2',
            show_results=True if train_results else False,
            results=zip(results['train_acc'],
                        results['train_loss'],
                        results['val_acc'],
                        results['val_loss'],
                        results['n_images'])), 200

    def post(self, epochs=None, batch_size=None, query=None):
        self.run_action(Action.TRAIN, train,
                        batch_size=batch_size,
                        epochs=epochs)
        flash('Training started!', 'success')
        return 202

    def delete(self):
        purge_json_file(self.cm.get_train_results_file(), EMPTY_TRAIN_RESULTS)
        return 200

    def _get_last_n_results(self, results, n, reverse):
        n = max(0, n)
        return {key: val[:n] for key, val in results.items()} if reverse \
            else {key: val[-n:] for key, val in results.items()}
