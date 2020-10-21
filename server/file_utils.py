import json
import os

from tqdm import tqdm

from dral.logger import LOG


def save_json(path, data):
    with open(path, "w+") as f:
        json.dump(data, f, indent=4)


def load_json(path):
    with open(path) as f:
        return json.load(f)


def load_last_predictions(path):
    predictions = load_json(path)
    return {int(key): val for key, val in predictions.items()}


def update_annotation_file(path, data_dir, label):
    try:
        annotations = load_json(path)
    except FileNotFoundError:
        annotations = {}
    annotations.setdefault(label, [])

    if os.path.isdir(data_dir):
        print(f'DATA DIR: {data_dir}')
        LOG.info(f'Start loading from {data_dir}...')
        for f in tqdm(os.listdir(data_dir)):
            feature_rpath = os.path.join(data_dir, f)
            annotations[label].append(feature_rpath)
    save_json(path, annotations)
    LOG.info(f'{path} file has been updated.')
