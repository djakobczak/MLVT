from server.views.base import BaseView
from server.extensions import executor
from server.actions.main import Action


class StatusView(BaseView):
    def get(self, action):
        status = executor.futures.done(Action(action))
        done = True if status is None else status
        return done, 200
