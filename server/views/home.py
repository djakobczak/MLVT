from connexion import request
from flask import render_template
from flask import redirect, url_for

from server.views.base import BaseView


class HomeView(BaseView):
    def search(self):
        return render_template('home.html.j2')

    def post(self):
        print("I am inside post")
        post_body = request.json
        print(post_body)
        return redirect(url_for('.views_HomeView_search'))
