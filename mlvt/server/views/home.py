from flask import render_template

from mlvt.server.views.base import BaseView
from mlvt.server.views.config import ConfigsView
from mlvt.server.file_utils import get_current_config


class HomeView(BaseView):
    def search(self):
        cv = ConfigsView()
        configs = cv.search()[0]
        current_config = get_current_config()
        return render_template('home.html.j2', configs=configs,
                               current_config=current_config)
