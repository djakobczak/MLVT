from mlvt.server.actions.main import ActionStatus
from mlvt.server.views.base import BaseView
from mlvt.server.extensions import executor
from mlvt.server.actions.main import Action


class StatusView(BaseView):
    def get(self, action):
        action = Action(action)
        finished = executor.futures.done(action)
        if finished is None:
            return {'status': ActionStatus.SUCCESS.value, 'data': ''}, 200

        status, data = executor.futures.pop(action).result() if \
            finished else (ActionStatus.ONGOING, 'Action is ongoing')
        return {'status': status.value, 'data': data}, 200
