from flask import jsonify, flash, render_template, \
    redirect, url_for

from dral.logger import LOG
from server.actions.main import Action
from server.views.base import ActionView
from server.actions.handlers import train
from server.file_utils import load_json, prune_json_file


class TrainView(ActionView):

    def search(self, number):
        train_results = load_json(self.cm.get_train_results_file(),
                                  parse_keys_to=int)

        if not train_results:
            flash('Training history is empty, '
                  'you have to train your model firstly', 'danger')
            return render_template(
                'train.html.j2', results=dict())

        number = len(train_results) - 1 if number < 0 else number
        try:
            result = train_results[number]
        except IndexError:
            flash(f'Training history contain only {len(train_results)}'
                  ' results, you got the most recent one', 'danger')
            result = train_results[number]

        return render_template(
            'train.html.j2',
            results=zip(result['acc'],
                        result['loss'])), 200

    def post(self, batch_size=None, query=None):
        self.run_action(Action.TRAIN, train)
        flash('Training started!', 'success')
        return 202

    def delete(self):
        prune_json_file(self.cm.get_train_results_file())
        return 200
