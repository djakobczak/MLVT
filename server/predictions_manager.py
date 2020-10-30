import numpy as np

from dral.logger import LOG
from server.file_utils import load_json, save_json


class PredictionsManager:
    def __init__(self, n_classes, predictions_path):
        self.n_classes = n_classes
        self.predictions_path = predictions_path

    def get_new_predictions(self, n_predictions,
                            model, dataloader, random, balance=True):
        LOG.info('Predict images.')
        print('[DEBUG] random: ', random)
        predictions, paths = model.predict_all(dataloader)
        print(len(predictions), len(paths))
        print(f'[DEBUG] {self.predictions_path}')
        self._save_predictions(predictions.tolist(), paths.tolist())
        return self._get_predictions(
            predictions, paths, n_predictions, random, balance)

    def get_predictions_from_file(self, n_predictions,
                                  random, balance=True):
        print('[DEBUG] random: ', random)
        LOG.info('Load predictions from file.')
        json_data = load_json(self.predictions_path)
        if not json_data:
            return {}

        predictions, paths = np.array(json_data.get('predictions')), \
            np.array(json_data.get('paths'))
        return self._get_predictions(
            predictions, paths, n_predictions, random, balance)

    def _get_predictions(self, predictions, paths, n_predictions,
                         random, balance=True):
        if random:
            return self._get_random(
                predictions, paths, n_predictions, balance)
        else:
            return self._get_most_uncertain(
                predictions, paths, n_predictions, balance)

    def remove_predictions(self, paths_to_remove):
        json_data = load_json(self.predictions_path)
        predictions, paths = np.array(json_data.get('predictions')), \
            np.array(json_data.get('paths'))
        idxs = [idx for idx, path in enumerate(paths)
                if path in paths_to_remove]
        print(paths)
        paths = np.delete(paths, idxs)
        predictions = np.delete(predictions, idxs, axis=0)
        predictions[[0, 1, 2]]
        self._save_predictions(predictions.tolist(), paths.tolist())

    def _save_predictions(self, predictions, paths):
        save_json(self.predictions_path, {
            'predictions': predictions,
            'paths': paths
        })

    def _get_most_uncertain(self, predictions, paths, n, balance=True):
        print(predictions)
        diffs = np.apply_along_axis(lambda x: np.absolute(x[0] - x[1]),
                                    1, predictions)
        labels = np.apply_along_axis(lambda x: np.argmax(x),
                                     1, predictions)
        idxs = np.argsort(diffs, axis=0)
        labels = labels[idxs]
        paths = paths[idxs]

        if balance:
            out_idxs = self._get_balanced_predictions(labels, n)
        else:
            out_idxs = range(2*n)

        out_labels = labels[out_idxs]
        out_paths = paths[out_idxs]
        out_mapping = {label: [] for label in range(self.n_classes)}
        for out_label, out_path in zip(out_labels, out_paths):
            out_mapping[out_label].append(out_path)

        return out_mapping

    def _get_random(self, predictions, paths, n, balance=True):
        labels = np.apply_along_axis(lambda x: np.argmax(x),
                                     1, predictions)
        if balance:
            out_idxs = self._get_balanced_predictions(labels, n)
        else:
            out_idxs = range(2*n)
        out_labels = labels[out_idxs]
        out_paths = paths[out_idxs]

        return self._create_mapping(out_labels, out_paths)

    def _get_balanced_predictions(self, labels, n):
        out_idx = []
        ctr = {label: 0 for label in range(self.n_classes)}
        for idx, label in enumerate(labels):
            if ctr[label] < n:
                out_idx.append(idx)
            if len(out_idx) >= 2 * n:
                return out_idx
            ctr[label] += 1
        return out_idx

    def _create_mapping(self, labels, paths):
        label_paths_mapping = {label: [] for label in range(self.n_classes)}
        for label, path in zip(labels, paths):
            label_paths_mapping[label].append(path)

        return label_paths_mapping
