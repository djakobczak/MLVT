from json import dumps


from flask import flash, render_template
import numpy as np
from plotly.utils import PlotlyJSONEncoder
import plotly.graph_objects as go
from plotly.subplots import make_subplots

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

GRAPH_WIDTH = 1400
GRAPH_HEIGHT = 600


class TrainView(ActionView):

    def search(self, nepochs, reverse):
        train_results = load_json(self.cm.get_train_results_file())
        if self.is_dict_empty(train_results):
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
        plot_acc = self._generate_acc_plot(train_acc, val_acc, n_images)
        plot_loss = self._generate_loss_plot(train_loss, val_loss, n_images)
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
        flash('Training started!', 'success')
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

    def generate_training_ploty(self):
        train_results = load_json(self.cm.get_train_results_file())
        train_acc, train_loss, val_acc, val_loss, n_images = \
            train_results['train_acc'], train_results['train_loss'], \
            train_results['val_acc'], train_results['val_loss'], \
            train_results['n_images']
        epochs = list(range(1, len(train_acc) + 1))
        # Create traces
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=epochs, y=train_acc,
                                 mode='lines',
                                 name='training'))
        fig.add_trace(go.Scatter(x=epochs, y=val_acc,
                                 mode='lines',
                                 name='validation'))

        fig.update_layout(
            title="Accuracy plot",
            xaxis_title="Epochs",
            yaxis_title="Accuracy",
            legend_title="Accuracy for data",
            width=1400,
            height=600,
        )
        fig.update_yaxes(range=[0, 1.0])
        figJSON = dumps(fig, cls=PlotlyJSONEncoder)
        return figJSON

    def _generate_acc_plot(self, train_acc, val_acc, n_images):
        epochs = list(range(1, len(train_acc) + 1))
        data = [
            (epochs, train_acc, "training acc", False, "lines"),
            (epochs, val_acc, "validation acc", False, "lines"),
            (epochs, n_images, "training images", True, "markers")
        ]

        fig = self._generate_scatter_plot(
            data=data,
            title="Accuracy plot",
            xaxis_title="Epochs",
            yaxis_title="Accuracy"
        )
        fig.update_yaxes(range=[0, 1.0], secondary_y=False)
        fig.update_yaxes(range=[0, max(n_images) + 5],
                         showgrid=False, secondary_y=True)
        fig.update_yaxes(title_text="Number of images",
                         secondary_y=True)
        figJSON = dumps(fig, cls=PlotlyJSONEncoder)
        return figJSON

    def _generate_loss_plot(self, train_loss, val_loss, n_images):
        epochs = list(range(1, len(train_loss) + 1))
        data = [
            (epochs, train_loss, "training loss", False, "lines"),
            (epochs, val_loss, "validation loss", False, "lines"),
            (epochs, n_images, "training images", True, "markers")
        ]

        fig = self._generate_scatter_plot(
            data=data,
            title="Loss plot",
            xaxis_title="Epochs",
            yaxis_title="Loss"
        )
        fig.update_yaxes(range=[0, max(max(train_loss), max(val_loss))*1.05],
                         secondary_y=False)
        fig.update_yaxes(range=[0, max(n_images) + 5],
                         showgrid=False, secondary_y=True)
        fig.update_yaxes(title_text="Number of images",
                         secondary_y=True)
        figJSON = dumps(fig, cls=PlotlyJSONEncoder)
        return figJSON

    def _generate_scatter_plot(self, data, title, xaxis_title, yaxis_title,
                               legend_title="Legend"):
        """Generate pyplot figure

        Args:
            data (tuple): (x, y, name, secondary_y, mode)
            title (str): plot title
            xaxis_title (str): x axis title
            yaxis_title (str): y axis title
            legend_title (str): legend title

        Returns:
            obj: pyplot serialized figure
        """
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        for x, y, name, secondary_y, mode in data:
            fig.add_trace(go.Scatter(x=x, y=y,
                                     mode=mode,
                                     name=name),
                          secondary_y=secondary_y)

        fig.update_layout(
            xaxis_title=xaxis_title,
            yaxis_title=yaxis_title,
            legend_title=legend_title,
            title={
                'text': title,
                'y': 0.9,
                'x': 0.5,
                'xanchor': 'center',
                'yanchor': 'top',
                'font': dict(
                    size=22
                )},
            margin=dict(l=70, r=10, t=100, b=60),
            paper_bgcolor='rgba(0,0,0,0)',
            font_color="rgba(230,230,230,255)"
            )
        return fig



    @staticmethod
    def is_dict_empty(d):
        for key, values in d.items():
            if values:
                return False
        return True
