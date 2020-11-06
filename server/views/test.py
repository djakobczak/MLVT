from flask import flash, render_template

from server.actions.handlers import test
from server.views.base import ActionView
from server.actions.main import Action
from server.file_utils import load_json, prune_json_file


class TestView(ActionView):
    def search(self, max_results):
        test_results = load_json(self.cm.get_test_results_file(),
                                 parse_keys_to=int)
        start_idx = len(test_results) - max_results
        results = [values for key, values in test_results.items()
                   if key > start_idx]
        if not results:
            flash('Test history is empty, '
                  'you have to test your model firstly', 'danger')
        return render_template('test.html.j2', results=results,
                               show_results=True if results else False), 200

    def post(self):
        self.run_action(Action.TEST, test)
        flash('Model evaluation started.', 'success')
        return 202

    def delete(self):
        prune_json_file(self.cm.get_test_results_file())
        return 200
