from mlvt.server.actions.main import ActionStatus, ActionDescription
from mlvt.server.views.base import BaseView
from mlvt.server.extensions import executor
from mlvt.server.actions.main import Action


class StatusView(BaseView):
    def get(self, action):
        action = Action(action)
        finished = executor.futures.done(action)
        if finished is None:
            return {'status': ActionStatus.SUCCESS.value,
                    'data': ActionDescription.SUCCESS.value}, 200

        status, data = executor.futures.pop(action).result() if finished \
            else (ActionStatus.ONGOING, ActionDescription.ONGOING.value)
        return {'status': status.value, 'data': data}, 200
