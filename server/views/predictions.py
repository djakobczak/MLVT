import json

from flask import render_template, url_for, request, redirect
import numpy as np

from dral.logger import LOG
from server.views.base import MLView
from server.file_utils import save_json, load_labels, label_samples
from server.exceptions import FileException
from server.action_lock import lock


class PredictionsView(MLView):

    @lock()
    def search(self, new_predictions, random, balance, maxImages):
        n_predictions = maxImages if maxImages else \
                            self.cm.get_number_of_predictions()
        if new_predictions:
            predictions = self._get_predictions(
                n_predictions, random=random, balance=balance)
            save_json(self.cm.get_last_predictions_file(), predictions)
        else:
            try:
                path = self.cm.get_last_predictions_file()
                predictions = load_labels(path)
                if not predictions:
                    predictions = self._get_predictions(
                        n_predictions, random=random, balance=balance)
                    save_json(self.cm.get_last_predictions_file(), predictions)
            except json.JSONDecodeError:
                raise FileException(
                    f"Server can not find file or it is corrupted: {path}")

        return render_template(
            "predictions.html.j2",
            path_start_idx=predictions[0][0].index('static'),  # html need realative path
            class1=predictions[0], class2=predictions[1],
            label1=self.cm.get_label_name(0),
            label2=self.cm.get_label_name(1)), 200

    def post(self):
        payload = request.json
        for class_num, (_, paths) in enumerate(payload.items()):
            label_samples(self.cm.get_unl_annotations_path(),
                          self.cm.get_train_annotations_path(),
                          paths, class_num)

        return redirect(url_for('.views_PredictionsView_search',
                                new_predictions=True))

    def _get_predictions(self, n_predictions, random=False, balance=True):
        LOG.info(f'Get {n_predictions} predictions with parameters: '
                 f'random={random}, balance={balance}')
        unl_loader = self.get_unl_loader()
        self.unl_dataset.load()
        model = self.load_model()
        predictions, paths = model.predict_all(unl_loader)
        print(f'Prediction done, load_time: {self.unl_dataset.load_time}, transform_time: {self.unl_dataset.trans_time}')

        if random:
            return self._get_random(
                predictions, paths, n_predictions)
        else:
            return self._get_most_uncertain(
                predictions, paths, n_predictions)

    def _get_most_uncertain(self, predictions, paths, n, balance=True):  # !TODO let it works for more then 2 classes
        labels = [np.argmin(el) for el in predictions]
        diffs = [abs(el[0] - el[1]) for el in predictions]
        paths_with_labels = [(path, label) for path, _, label
                             in sorted(zip(paths, diffs, labels),
                             key=lambda pair: pair[1])]
        return self._create_mapping(paths_with_labels, n, balance)

    def _get_random(self, predictions, paths, n, balance=True):
        labels = [np.argmin(el) for el in predictions]
        return self._create_mapping(list(zip(paths, labels)), n, balance)

    def _create_mapping(self, paths_with_labels, n, balance=True):
        """Get tuple (paths, labels) and create dictionary label: array of
        apths.

        Args:
            paths_with_labels (tuple): paths corresponding labels tuple
            n (integer): number of returned items
            balance (bool, optional): If set to True limit number of returned
            paths of one class to n // 2. Defaults to True.

        Returns:
            dict: label: paths dictionary
        """
        n = n if not n % 2 else n - 1  # n should be even
        label_paths_mapping = {label: []
                               for label in self.cm.get_numeric_labels()}
        for path, label in paths_with_labels:
            if balance:
                if len(label_paths_mapping[label]) < n // 2:
                    label_paths_mapping[label].append(path)
            else:
                label_paths_mapping[label].append(path)

            if sum(len(arr) for arr in label_paths_mapping.values()) >= n:
                break
        print(label_paths_mapping)
        return label_paths_mapping
