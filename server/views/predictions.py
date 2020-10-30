import json

from flask import render_template, url_for, request, redirect

from server.actions.main import Action
from server.actions.handlers import predict
from server.views.base import ActionView
from server.file_utils import label_samples
from server.exceptions import FileException
from server.predictions_manager import PredictionsManager


class PredictionsView(ActionView):
    def search(self, new_predictions, random, balance, maxImages=None):
        n_predictions = maxImages if maxImages else \
            self.cm.get_number_of_predictions()

        try:
            pm = PredictionsManager(self.cm.get_n_labels(),
                                    self.cm.get_predictions_file())
            predictions = pm.get_predictions_from_file(
                n_predictions, random, balance)
        except json.JSONDecodeError:
            raise FileException(
                f"Server can not find file or it is corrupted")

        if new_predictions:   # or not predictions
            self.run_action(Action.PREDICTION, predict,
                            n_predictions=n_predictions,
                            random=random, balance=balance)

        idx = predictions[0][0].index('static') if predictions else 0
        return render_template(
            "predictions.html.j2",
            path_start_idx=idx,  # html need realative path
            class1=predictions.get(0, []), class2=predictions.get(1, []),
            label1=self.cm.get_label_name(0),
            label2=self.cm.get_label_name(1),
            n_images=self.cm.get_n_predictions()), 200

    def post(self):
        payload = request.json
        for class_num, (_, paths) in enumerate(payload.items()):
            pm = PredictionsManager(self.cm.get_n_labels(),
                                    self.cm.get_predictions_file())
            label_samples(self.cm.get_unl_annotations_path(),
                          self.cm.get_train_annotations_path(),
                          paths, class_num, pm)

        return 200
