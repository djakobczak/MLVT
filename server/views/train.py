from flask import jsonify, flash, render_template, \
    redirect, url_for

from dral.logger import LOG
from server.actions.main import Action
from server.views.base import MLView
from server.action_lock import lock
from server.actions.train import train
from server.extensions import executor
from server.exceptions import ActionLockedException
from server.file_utils import load_json


# !TODO add context manager for load/save flow
class TrainView(MLView):

    def search(self, number):
        train_results = load_json(self.cm.get_train_results_file(),
                                  parse_keys_to=float)

        if not len(train_results):
            flash('Training history is empty, '
                  'you have to train your model firstly', 'danger')
            return render_template(
                'train.html.j2', results=dict())

        try:
            timestamp = sorted(train_results.keys(), reverse=True)[number]
        except IndexError:
            flash(f'Training history contain only {len(train_results)}'
                  ' results, you got the most recent one', 'danger')
            timestamp = sorted(train_results.keys(), reverse=True)[-1]
        print(list(train_results[timestamp].items()))
        return render_template(
            'train.html.j2',
            results=zip(train_results[timestamp]['acc'],
                        train_results[timestamp]['loss']))

    def post(self, batch_size=None, query=None):
        if Action.TRAIN in executor.futures._futures:
            if not executor.futures.done(Action.TRAIN):
                raise ActionLockedException("Ongoing action!")
            action_result = executor.futures.pop(Action.TRAIN)  # !TODO handle action result

        executor.submit_stored(Action.TRAIN, train)
        flash('Traing started!', 'primary')
        return 202
