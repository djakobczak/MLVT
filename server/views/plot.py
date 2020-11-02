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
        train_results = load_json(self.cm.get_train_results_file(),
                                  parse_keys_to=int)

        result = train_results[len(train_results) - 1]
        training_acc = result['acc']
        validation_acc = result['validation_acc']

        range_ = len(training_acc) + 1
        max_acc = 1.1
        x = range(1, range_)

        plt.style.use('seaborn')
        fig, ax1 = plt.subplots()
        ax1.set_xlim([0, range_])
        ax1.set_ylim([0, max_acc])
        ax1.set_xticks(range(range_))
        ax1.set_yticks(np.linspace(0, max_acc, int(max_acc*10)+1))

        ax1.plot(x, training_acc, 'r--', x, validation_acc, 'g-')
        ax1.grid(True, axis='y')
        ax1.set_xlabel('Epochs')
        ax1.set_ylabel('Accuracy')
        ax1.legend(['Training', 'Validation'])
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
