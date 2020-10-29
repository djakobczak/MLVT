import json
import os
from pathlib import Path

from tqdm import tqdm

from dral.logger import LOG
from server.exceptions import PathException


def save_json(path, data):
    with open(path, "w") as f:
        print('save!!')
        json.dump(data, f, indent=4, sort_keys=True)


def load_json(path, parse_keys_to=None):  # !TODO change load_labels to laod_json with parse
    try:
        with open(path) as f:
            dict_data = json.load(f)
            if parse_keys_to:
                return {parse_keys_to(key): val
                        for key, val in dict_data.items()}
            return dict_data
    except FileNotFoundError:
        return {}


def get_last_n_images_key(path):
    train_results = load_json(path, parse_keys_to=int)
    if not train_results:
        return 0
    last_result = train_results[len(train_results) - 1]
    return last_result['n_images']


def load_labels(path):
    mapping = load_json(path)
    return {int(key): val for key, val in mapping.items()}


def append_to_json_file(path, values):
    dict_data = load_json(path, parse_keys_to=int)
    dict_data[len(dict_data)] = values
    save_json(path, dict_data)


def is_json_empty(path):  # path must contain iterable keys
    """Returns sum of the elements in json file specified by path

    Args:
        path (str): path to json file

    Returns:
        int: sum of all values lengths
    """
    dict_ = load_json(path)
    return sum(len(dict_[value]) for value in dict_)


def update_annotation_file(path, data_dir, label):
    try:
        annotations = load_json(path, parse_keys_to=int)
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


def prune_json_file(path):
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
