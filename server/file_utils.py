import json
import os
from pathlib import Path

from tqdm import tqdm

from dral.logger import LOG
from server.exceptions import PathException


def save_json(path, data):
    with open(path, "w+") as f:
        print('[DEBUG] save')
        json.dump(data, f, indent=4, sort_keys=True)


def load_json(path):
    with open(path) as f:
        return json.load(f)


def load_labels(path):
    mapping = load_json(path)
    return {int(key): val for key, val in mapping.items()}


def update_annotation_file(path, data_dir, label, prune=False):
    try:
        annotations = load_labels(path)
    except FileNotFoundError:
        annotations = {}
    annotations.setdefault(label, [])

    if os.path.isdir(data_dir):
        LOG.info(f'Start loading from {data_dir}...')
        for f in tqdm(os.listdir(data_dir)):
            feature_rpath = os.path.join(data_dir, f)
            annotations[label].append(feature_rpath)
    save_json(path, annotations)
    LOG.info(f'{path} file has been updated.')


def prune_annotation_file(path):
    LOG.info(f'Prune file: {path}')
    save_json(path, {})


def label_samples(unl_json_file, label_json_file, paths,
                  label, unl_label=255):
    unl_json = load_labels(unl_json_file)
    label_json = load_labels(label_json_file)
    unl_samples = unl_json[unl_label]
    labelled_list = label_json[label]
    for path in paths:
        unl_samples.remove(path)
        labelled_list.append(path)
    save_json(unl_json_file, unl_json)
    save_json(label_json_file, label_json)


def create_subdirs_if_not_exist(path):
    head, tail = os.path.split(path)
    Path(head).mkdir(parents=True, exist_ok=True)


def fail_if_headpath_not_exist(path):
    head, tail = os.path.split(path)
    if not os.path.exists(head):
        raise PathException(f'({path}) does not exist.')