import json
import os
from pathlib import Path
import shutil

from tqdm import tqdm

from mlvt.model.logger import LOG
from mlvt.server.exceptions import PathException


def save_json(path, data, force=True):
    if force:
        create_subdirs_if_not_exist(path)
    with open(path, "w") as f:
        LOG.info(f'save json at path: {path}')
        json.dump(data, f, indent=4, sort_keys=True)


def load_json(path, parse_keys_to=None):
    try:
        with open(path) as f:
            dict_data = json.load(f)
            if parse_keys_to:
                return {parse_keys_to(key): val
                        for key, val in dict_data.items()}
            return dict_data
    except FileNotFoundError:
        LOG.warning(f"File ({path}) not found")
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


def append_to_train_file(path, values):
    LOG.info("Append new training data")
    dict_data = load_json(path)

    dict_data['train_acc'].extend(values['train_acc'])
    dict_data['train_loss'].extend(values['train_loss'])
    dict_data['val_acc'].extend(values['val_acc'])
    dict_data['val_loss'].extend(values['val_loss'])
    dict_data['n_images'].extend(values['n_images'])
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


def purge_json_file(path, content=None):
    if content is None:
        content = {}
    LOG.info(f'Purge file: {path}')
    save_json(path, content)


def clear_dir(dir_path):
    for filename in os.listdir(dir_path):
        file_path = os.path.join(dir_path, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            LOG.error('Failed to delete %s. Reason: %s' % (file_path, e))


def label_samples(unl_json_file, label_json_file, paths,
                  label, pm, unl_label=255):
    unl_json = load_labels(unl_json_file)
    label_json = load_labels(label_json_file)
    unl_samples = unl_json[unl_label]
    labelled_list = label_json[label]
    pm.remove_predictions(paths)
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


def is_dict_empty(d):
    for key, values in d.items():
        if values:
            return False
    return True
