from connexion import request
from flask import render_template
from flask import redirect, url_for

from mlvt.server.views.base import BaseView


class HomeView(BaseView):
    def search(self):
        return render_template('home.html.j2')

    def post(self):
        return redirect(url_for('.views_HomeView_search'))
