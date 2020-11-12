from enum import Enum
import io

from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import matplotlib.pyplot as plt
import numpy as np
from flask import Response

from server.views.base import BaseView
from server.file_utils import load_json


class PlotType(Enum):
    TRAINING_ACC = 'train_acc'
    TEST = 'test'


class PlotView(BaseView):
    def search(self, plot_type):
        if plot_type == PlotType.TRAINING_ACC.value:
            fig = self.generate_training_acc_plot()
        elif plot_type == PlotType.TEST.value:
            fig = self.generate_test_plot()

        output = io.BytesIO()
        FigureCanvas(fig).print_png(output)
        return Response(output.getvalue(), mimetype='image/png')

    def generate_training_acc_plot(self):
        train_results = load_json(self.cm.get_train_results_file())
        train_acc, train_loss, val_acc, val_loss, n_images = \
            train_results['train_acc'], train_results['train_loss'], \
            train_results['val_acc'], train_results['val_loss'], \
            train_results['n_images']
        range_ = len(train_acc) + 1
        max_acc = 1.1
        max_imgs = int(max(n_images) * 1.05)
        max_loss = max(max(val_loss), max(train_loss)) * 1.05
        min_loss = min(min(val_loss), min(train_loss)) * 0.95
        epochs = range(1, range_)
        yticks_points = 12

        plt.style.use('seaborn')
        fig, ax = plt.subplots(2, figsize=(14, 8))
        fig.subplots_adjust(left=0.09, right=0.92, top=0.9, bottom=0.1)
        twin_ax = []
        twin_ax.append(ax[0].twinx())
        twin_ax.append(ax[1].twinx())

        ax[0].set_xlim([0, range_])
        ax[1].set_xlim([0, range_])
        ax[0].set_ylim([0, max_acc])
        twin_ax[0].set_ylim([0, max_imgs])
        ax[1].set_ylim([min_loss, max_loss])
        twin_ax[1].set_ylim([0, max_imgs])
        ax[0].set_xticks(self._get_xticks(len(epochs)))
        ax[1].set_xticks(self._get_xticks(len(epochs)))
        ax[0].set_yticks(np.linspace(0, max_acc, yticks_points))
        ax[1].set_yticks(np.around(np.linspace(
            min_loss, max_loss, yticks_points), decimals=3))

        # set plots
        tacc_plot = ax[0].plot(epochs, train_acc, 'b--',
                               label='Train accuracy')
        vacc_plot = ax[0].plot(epochs, val_acc, 'g-',
                               label='Validation accuracy')
        nimg_plot1 = twin_ax[0].plot(epochs, n_images, '.',
                                     color='salmon', label="Number of images")
        tloss_plot = ax[1].plot(epochs, train_loss, 'b--',
                                label='Train loss')
        vloss_plot = ax[1].plot(epochs, val_loss, 'g-',
                                label='Validation loss')
        nimg_plot2 = twin_ax[1].plot(epochs, n_images, '.',
                                     color='salmon', label="Number of images")
        ax[0].grid(True)
        twin_ax[0].grid(False)
        ax[1].grid(True)
        twin_ax[1].grid(False)
        ax[0].set_xlabel('Epochs')
        ax[1].set_xlabel('Epochs')
        ax[0].set_ylabel('Accuracy')
        twin_ax[0].set_ylabel('Number of training images')
        ax[1].set_ylabel('Loss')
        twin_ax[1].set_ylabel('Number of training images')

        plots1 = tacc_plot + vacc_plot + nimg_plot1
        labs1 = [p.get_label() for p in plots1]
        ax[0].legend(plots1, labs1, loc="lower right")

        plots2 = tloss_plot + vloss_plot + nimg_plot2
        labs2 = [p.get_label() for p in plots2]
        ax[1].legend(plots2, labs2, loc="upper right",
                     bbox_to_anchor=(1, 0.92))
        return fig

    def generate_test_plot(self, max_results=20):
        test_results = load_json(self.cm.get_test_results_file(),
                                 parse_keys_to=int)
        start_idx = len(test_results) - max_results
        accs = [values['acc'] for key, values in test_results.items()
                if key > start_idx]
        n_images = [values['n_images'] for key, values in test_results.items()
                    if key > start_idx]
        max_imgs = max(n_images)
        test_idx = len(accs) + 1

        max_acc = 1.0
        x = range(1, test_idx)

        plt.style.use('seaborn-dark')
        fig, ax1 = plt.subplots()
        ax2 = ax1.twinx()

        ax1.set_xlim([0, test_idx])
        ax1.set_ylim([0, max_acc])
        ax1.set_xticks(range(test_idx))
        ax1.set_yticks(np.linspace(0, max_acc, int(max_acc*10)+1))

        plot1 = ax1.plot(x, accs, 'g-', label='Accuracy')
        ax1.grid(True, axis='y')
        ax1.set_xlabel('History')
        ax1.set_ylabel('Accuracy')
        # ax1.legend(['Test accuracy'])

        plot2 = ax2.plot(x, n_images, 'b--', label='Number of images')
        ax2.set_ylim([0, int(max_imgs*1.1)])
        ax2.set_ylabel('Number of training images')
        # ax2.legend(['Training images'], loc="upper left")

        plots = plot1 + plot2
        labs = [p.get_label() for p in plots]
        ax1.legend(plots, labs, loc="lower right")
        return fig

    def _concat_training_sessions(self, train_results, start_idx=0):
        acc = []
        loss = []
        val_acc = []
        val_loss = []
        n_images = []
        for idx in range(start_idx, len(train_results)):
            result = train_results[idx]
            acc.extend(result['accs'])
            loss.extend(result['losses'])
            val_acc.extend(result['val_accs'])
            val_loss.extend(result['val_losses'])
            n_images.extend([result['n_images']] * len(result['accs']))
        return (acc, loss, val_acc, val_loss, n_images)

    def _get_xticks(self, nepochs, max_xticks=21):
        if max_xticks >= nepochs:
            return range(nepochs+1)
        step = nepochs // max_xticks + 1
        print(step)
        return range(0, nepochs+1, step)
