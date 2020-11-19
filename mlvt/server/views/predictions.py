import json

from flask import render_template, request

from mlvt.server.actions.main import Action
from mlvt.server.actions.handlers import predict
from mlvt.server.config import CUT_STATIC_IDX
from mlvt.server.exceptions import FileException
from mlvt.server.views.annotation import AnnotationsView
from mlvt.server.views.base import ActionView
from mlvt.server.file_utils import label_samples
from mlvt.server.predictions_manager import PredictionsManager
from mlvt.server.utils import DatasetType


class PredictionsView(ActionView):
    def search(self, new_predictions, random, balance, maxImages=None):
        self.init_cm()
        n_predictions = maxImages if maxImages else \
            self.cm.get_number_of_predictions()

        try:
            pm = PredictionsManager(self.cm.get_n_labels(),
                                    self.cm.get_predictions_file())
            predictions = pm.get_predictions_from_file(
                n_predictions, random, balance)
        except json.JSONDecodeError:
            raise FileException("Server can not find file or it is corrupted")

        if new_predictions:
            self.run_action(Action.PREDICTION, predict,
                            n_predictions=n_predictions,
                            random=random, balance=balance)
        av = AnnotationsView()
        return render_template(
            "predictions.html.j2",
            path_start_idx=CUT_STATIC_IDX,  # html need realative path
            class1=predictions.get(0, []), class2=predictions.get(1, []),
            label1=self.cm.get_label_name(0),
            label2=self.cm.get_label_name(1),
            n_images=self.cm.get_n_predictions(),
            train_summary=av.get(DatasetType.TRAIN.value)[0],
            unl_summary=av.get(DatasetType.UNLABELLED.value)[0]), 200

    def post(self):
        self.init_cm()
        payload = request.json
        for class_num, (_, paths) in enumerate(payload.items()):
            pm = PredictionsManager(self.cm.get_n_labels(),
                                    self.cm.get_predictions_file())
            label_samples(self.cm.get_unl_annotations_path(),
                          self.cm.get_train_annotations_path(),
                          paths, class_num, pm)

        return '', 200
