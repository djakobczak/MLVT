from flask import flash, render_template
import numpy as np

from mlvt.server.actions.handlers import train
from mlvt.server.actions.main import Action
from mlvt.server.views.base import ActionView
from mlvt.server.plots import Plot
from mlvt.server.file_utils import load_json, purge_json_file, \
    is_dict_empty


EMPTY_TRAIN_RESULTS = {
    'train_acc': [], 'train_loss': [],
    'val_acc': [], 'val_loss': [],
    'n_images': []
}


class TrainView(ActionView):

    def search(self, nepochs, reverse):
        train_results = load_json(self.cm.get_train_results_file())
        if is_dict_empty(train_results):
            flash('Training history is empty, '
                  'you have to train your model firstly', 'danger')
            return render_template(
                'train.html.j2', results=dict(),
                show_results=False,
                default_epochs=self.cm.get_epochs(),
                default_bs=self.cm.get_batch_size())

        results = self._get_last_n_results(train_results, nepochs, reverse)
        train_acc, train_loss, val_acc, val_loss, n_images = \
            train_results['train_acc'], train_results['train_loss'], \
            train_results['val_acc'], train_results['val_loss'], \
            train_results['n_images']
        stats = self._get_training_stats(results)
        plot = Plot()
        plot_acc = plot.generate_acc_plot(train_acc, val_acc, n_images)
        plot_loss = plot.generate_loss_plot(train_loss, val_loss, n_images)
        return render_template(
            'train.html.j2',
            show_results=True,
            stats=stats,
            plot_acc=plot_acc,
            plot_loss=plot_loss,
            default_epochs=self.cm.get_epochs(),
            default_bs=self.cm.get_batch_size(),
            results=zip(results['train_acc'],
                        results['train_loss'],
                        results['val_acc'],
                        results['val_loss'],
                        results['n_images'])), 200

    def post(self, epochs=None, batch_size=None, query=None):
        self.run_action(Action.TRAIN, train,
                        batch_size=batch_size,
                        epochs=epochs)
        return 202

    def delete(self):
        purge_json_file(self.cm.get_train_results_file(), EMPTY_TRAIN_RESULTS)
        return 200

    def _get_last_n_results(self, results, n, reverse):
        n = max(0, n)
        return {key: val[:n] for key, val in results.items()} if reverse \
            else {key: val[-n:] for key, val in results.items()}

    def _get_training_stats(self, results):
        tacc = results.get('train_acc')
        tacc_epoch = np.argmax(tacc)
        tacc = tacc[tacc_epoch]
        tacc_description = \
            f'Maximum training accuracy achieved on {tacc_epoch + 1} epoch'

        tloss = results.get('train_loss')
        tloss_epoch = np.argmin(tloss)
        tloss = tloss[tloss_epoch]
        tloss_description = \
            f'Minimum training loss achieved on {tloss_epoch + 1} epoch'

        vacc = results.get('val_acc')
        vacc_epoch = np.argmax(vacc)
        vacc = vacc[vacc_epoch]
        vacc_description = \
            f'Maximum validation accuracy achieved on {vacc_epoch + 1} epoch'

        vloss = results.get('val_loss')
        vloss_epoch = np.argmin(vloss)
        vloss = vloss[vloss_epoch]
        vloss_description = \
            f'Minimum validation loss achieved on {vloss_epoch + 1} epoch'

        return {
            'tacc': tacc,
            'tacc_epoch': tacc_epoch + 1,
            'tacc_description': tacc_description,
            'tloss': tloss,
            'tloss_epoch': tloss_epoch + 1,
            'tloss_description': tloss_description,
            'vacc': vacc,
            'vacc_epoch': vacc_epoch + 1,
            'vacc_description': vacc_description,
            'vloss': vloss,
            'vloss_epoch': vloss_epoch + 1,
            'vloss_description': vloss_description,
        }
