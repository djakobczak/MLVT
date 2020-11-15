from json import dumps

from plotly.utils import PlotlyJSONEncoder
import plotly.graph_objects as go
from plotly.subplots import make_subplots


class Plot:
    def generate_acc_plot(self, train_acc, val_acc, n_images):
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

    def generate_loss_plot(self, train_loss, val_loss, n_images):
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
        """Generate plotly figure

        Args:
            data (tuple): (x, y, name, secondary_y, mode)
            title (str): plot title
            xaxis_title (str): x axis title
            yaxis_title (str): y axis title
            legend_title (str): legend title

        Returns:
            obj: plotly serialized figure
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
