from functools import wraps
import os

import numpy as np
import confuse


CONFIG_PATH = CONFIG_PATH = os.path.join(
    '.', 'mlvt', 'model', 'config', 'config.yml')

config = confuse.Configuration('DRAL', __name__)
config.set_file(CONFIG_PATH)  # !TODO change location of file path


class ConfigManager:
    configurations = config.keys()

    def __init__(self, config_name):
        if config_name not in self.configurations:
            raise ValueError(f'({config_name}) is not defined in config file. '
                             f'Defined configurations: {self.configurations}')

        self.config = config[config_name]
        self.config_name = config_name
        self.preprocessing = 'preprocessing'

    def get_config(self):
        return self.config.get()

    def get_dataset_name(self):
        return self.config['dataset'].get(str)

    def get_img_size(self):
        return self.config['general']['image_size'].get(int)

    def get_n_labels(self):
        return len(self.config['general']['label_mapping'].get(dict))

    def get_label_mapping(self, idx=None):
        return self.config['general']['label_mapping'].get(dict)

    def get_label_names(self):
        return list(self.get_label_mapping().keys())

    def get_label_name(self, idx):
        return self.get_label_names()[idx]

    def get_number_of_predictions(self):
        return self.config['train']['predictions'].get(int)

    def do_shuffle(self):
        return self.config['loader']['shuffle'].get(bool)

    def get_label_format(self):
        return self.config['loader']['label_format'].get(str)

    def do_grayscale(self):
        return self.config[self.preprocessing]['grayscale'].get(bool)

    def do_rescale_with_crop(self):
        return self.config[self.preprocessing]['rescale_with_crop'].get(bool)

    def do_normalization(self):
        return self.config[self.preprocessing]['normalization'].get(bool)

    def do_centering(self):
        return self.config[self.preprocessing]['centering'].get(bool)

    def do_standarization(self):
        return self.config[self.preprocessing]['standarization'].get(bool)

    def do_strict_balance(self):
        return self.config[self.preprocessing]['strict_balance'].get(bool)

    def get_config_name(self):
        return self.config_name

    def get_tmp_dir(self):
        return 'tmp'

    def get_unknown_label(self):
        return self.config['general']['unknown'].get(int)

    def get_one_hot_labels(self):
        labels = []
        for k, label in enumerate(self.get_label_names()):
            label = np.eye(self.get_number_of_labels())[k]
            labels.append(label)
        return labels

    def get_numeric_labels(self):
        return list(self.get_label_mapping().values())

    def enable_npy_preprocessing(self, enable):
        """Switch config between numpy config and pure images config

        Args:
            enable (bool): if set to True then use numpy config
            for preprocessing
        """
        self.preprocessing = 'preprocessing_npy' if enable else 'preprocessing'

    # base server path
    def _get_base_server_path(self):
        return self.config['paths']['server_base'].get(list)

    def _get_base_data_path(self):
        return self.config['paths']['data_base'].get(list)

    def server_path(join_multiple=False):
        """Decorator that adds server static path prefix and
         create os independent path.

        Args:
            f (obj): function that returns path in list format
            e.g. [path, to, file]

        Returns:
            str: path
        """
        def decorator(f):
            @wraps(f)
            def inner(*args, **kwargs):
                out = f(*args, **kwargs)
                if join_multiple:
                    return [os.path.join(*args[0]._get_base_server_path(),
                            *path) for path in out]
                return os.path.join(*args[0]._get_base_server_path(), *out)
            return inner
        return decorator

    def data_path(join_multiple=False):
        """Decorator that adds base data path prefix and
        create os independent path.

        Args:
            f (obj): function that returns path in list format
            e.g. [path, to, file]

        Returns:
            str: path
        """
        def decorator(f):
            @wraps(f)
            def inner(*args, **kwargs):
                out = f(*args, **kwargs)
                if join_multiple:
                    return [os.path.join(*args[0]._get_base_data_path(),
                            *path) for path in out]
                return os.path.join(*args[0]._get_base_data_path(), *out)
            return inner
        return decorator

    # annotation files paths
    @server_path()
    def get_annotation_dir(self):
        return self.config['paths']['annotations']['dir'].get(list)

    def get_unl_annotations_path(self):
        return os.path.join(
            self.get_annotation_dir(),
            self.get_unl_annotations_filename())

    def get_unl_annotations_filename(self):
        return self.config['paths']['annotations']['unl'].get(str)

    def get_train_annotations_path(self):
        return os.path.join(
            self.get_annotation_dir(),
            self.get_train_annotations_filename())

    def get_train_annotations_filename(self):
        return self.config['paths']['annotations']['train'].get(str)

    def get_validation_annotations_path(self):
        return os.path.join(
            self.get_annotation_dir(),
            self.get_validation_annotations_filename())

    def get_validation_annotations_filename(self):
        return self.config['paths']['annotations']['valid'].get(str)

    def get_test_annotations_path(self):
        return os.path.join(
            self.get_annotation_dir(),
            self.get_test_annotations_filename())

    def get_test_annotations_filename(self):
        return self.config['paths']['annotations']['test'].get(str)

    # model paths
    @server_path()
    def get_model_raw(self):
        return self.config['paths']['models']['raw'].get(list)

    @server_path()
    def get_model_trained(self):
        return self.config['paths']['models']['trained'].get(list)

    def _raw_images_path(self):
        return self.config['paths']['images']['raw']

    def _transformed_images_path(self):
        return self.config['paths']['images']['transformed']

    # raw paths
    @data_path()
    def get_raw_unl_dir(self):
        return self._raw_images_path()['unl'].get(list)

    @data_path(join_multiple=True)
    def get_raw_train_dirs(self):
        return self._join_with_clssses(
            self._raw_images_path()['train'].get(list))

    @data_path(join_multiple=True)
    def get_raw_test_dirs(self):
        return self._join_with_clssses(
            self._raw_images_path()['test'].get(list))

    @data_path(join_multiple=True)
    def get_raw_validation_dirs(self):
        return self._join_with_clssses(
            self._raw_images_path()['validation'].get(list))

    # transformed dirs
    @server_path()
    def get_unl_transformed_dir(self):
        return self._transformed_images_path()['unl'].get(list)

    @server_path(join_multiple=True)
    def get_train_transformed_dirs(self):
        return self._join_with_clssses(
            self._transformed_images_path()['train'].get(list))

    @server_path(join_multiple=True)
    def get_test_transformed_dirs(self):
        return self._join_with_clssses(
            self._transformed_images_path()['test'].get(list))

    @server_path(join_multiple=True)
    def get_validation_transformed_dirs(self):
        return self._join_with_clssses(
            self._transformed_images_path()['validation'].get(list))

    def _join_with_clssses(self, path):
        return [path + [class_name] for class_name in self.get_label_names()]

    # data paths
    @server_path()
    def get_train_results_file(self):
        return self.config['paths']['data']['train_results'].get(list)

    @server_path()
    def get_test_results_file(self):
        return self.config['paths']['data']['test_results'].get(list)

    @server_path()
    def get_predictions_file(self):
        return self.config['paths']['data']['predictions'].get(list)

    @server_path()
    def get_last_user_test_path(self):
        return self.config['paths']['data']['last_user_test'].get(list)

    # train
    def get_batch_size(self):
        return self.config['train']['batch_size'].get(int)

    def get_epochs(self):
        return self.config['train']['epochs'].get(int)

    def get_n_predictions(self):
        return self.config['train']['predictions'].get(int)

    # test
    def get_test_n_outputs(self):
        return self.config['test']['outputs'].get(int)
