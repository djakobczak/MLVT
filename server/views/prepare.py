from flask import render_template

from server.views.base import BaseView


class PrepareView(BaseView):
    def search(self):
        return render_template("prepare.html.j2")
