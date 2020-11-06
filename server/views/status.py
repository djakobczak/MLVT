from server.actions.main import ActionStatus
from server.views.base import BaseView
from server.extensions import executor
from server.actions.main import Action


class StatusView(BaseView):
    def get(self, action):
        action = Action(action)
        finished = executor.futures.done(action)
        if finished is None:
            return {'status': ActionStatus.SUCCESS.value, 'data': ''}, 200

        status, data = executor.futures.pop(action).result() if \
            finished else (ActionStatus.ONGOING, 'Action is ongoing')
        return {'status': status.value, 'data': data}, 200
