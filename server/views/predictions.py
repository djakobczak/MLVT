from flask import render_template, url_for, request, redirect
import numpy as np

from dral.annotations import label_samples
from dral.utils import get_relative_paths
from server.views.base import MLView


class PredictionsView(MLView):

    def search(self):
        random = request.args.get('random')
        balance = request.args.get('balance')
        n_predictions = int(request.args.get('maxImages',
                            self.cm.get_number_of_predictions()))

        paths = self._get_predictions(
            n_predictions, random=random, balance=balance)
        return render_template("predictions.html.j2",
                               path_start_idx=paths[0][0].index('static'),  # html need realative path
                               class1=paths[0], class2=paths[1],
                               label1=self.cm.get_label_name(0),
                               label2=self.cm.get_label_name(1)), 200

    def post(self):
        payload = request.json
        for class_num, (_, paths) in enumerate(payload.items()):
            label_samples(self.cm.get_unl_annotations_path(),
                          self.cm.get_train_annotations_path(),
                          paths, class_num)

        return "success", 200

    def _get_predictions(self, n_predictions, random=False, balance=True):
        unl_loader = self.get_unl_loader()
        model = self.load_model()
        print(len(self.unl_dataset))
        print(len(unl_loader))
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

        return label_paths_mapping
