import hashlib

from dral.logger import LOG
from server.file_utils import load_json
from server.plots import Plot
from server.views.base import BaseView
from server.utils import last_checksum
from server.file_utils import is_dict_empty
from server.exceptions import EmptyFileException


class PlotView(BaseView):
    def search(self, force):
        path = self.cm.get_train_results_file()
        if force:
            return self._get_plotly_graphs(path), 200

        checksum = hashlib.md5(open(path).read().encode("utf-8")).hexdigest()
        LOG.info(f"Last checksum: {last_checksum.value}, "
                 f"current checksum: {checksum}")
        if last_checksum.value == checksum:
            return '', 204

        last_checksum.value = checksum
        print('RETURN GRAPH')
        return self._get_plotly_graphs(path), 200

    def _get_plotly_graphs(self, path):
        train_results = load_json(path)
        if is_dict_empty(train_results):
            raise EmptyFileException("No training results")

        train_acc, train_loss, val_acc, val_loss, n_images = \
            train_results['train_acc'], train_results['train_loss'], \
            train_results['val_acc'], train_results['val_loss'], \
            train_results['n_images']

        plot = Plot()
        plot_acc = plot.generate_acc_plot(train_acc, val_acc, n_images)
        plot_loss = plot.generate_loss_plot(train_loss, val_loss, n_images)
        return {"plot_acc": plot_acc, "plot_loss": plot_loss}
