import os

import numpy as np
import confuse

CONFIG_PATH = 'dral/config/config.yml'
UNKNOWN_LABEL = 'Unknown'

config = confuse.Configuration('DRAL', __name__)
config.set_file(CONFIG_PATH)  # !TODO change location of file path

DEFAULT_CONFIG = 'testset'


# !TODO should be splitted into sections e.g. cm.train.batch_size.get()
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

    # annotation files paths
    def get_annotation_dir(self):
        return os.path.join(
            *self.config['paths']['annotations']['dir'].get(str).split('/'))

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

    def get_test_annotations_path(self):
        return os.path.join(
            self.get_annotation_dir(),
            self.get_test_annotations_filename())

    def get_test_annotations_filename(self):
        return self.config['paths']['annotations']['test'].get(str)

    # model paths
    def get_model_raw(self):
        return os.path.join(
            *self.config['paths']['models']['raw'].get(str).split('/'))

    def get_model_trained(self):
        return os.path.join(
            *self.config['paths']['models']['trained'].get(str).split('/'))

    # raw paths
    def get_raw_unl_dir(self):
        return os.path.join(
            *self.config['paths']['images']['raw_unl'].get(str).split('/'))

    def get_raw_train_dirs(self):
        return self._join_with_clssses(
            *self.config['paths']['images']['raw_train'].
            get(str).split('/'))

    def get_raw_test_dirs(self):
        return self._join_with_clssses(
            *self.config['paths']['images']['raw_test'].
            get(str).split('/'))

    def get_raw_validation_dirs(self):
        return self._join_with_clssses(
            *self.config['paths']['images']['raw_validation'].
            get(str).split('/'))

    # transformed dirs
    def get_unl_transformed_dir(self):
        return os.path.join(
            *self.config['paths']['images']['transformed_unl'].
            get(str).split('/'))

    def get_train_transformed_dirs(self):
        return self._join_with_clssses(
            *self.config['paths']['images']['transformed_train'].
            get(str).split('/'))

    def get_test_transformed_dirs(self):
        return self._join_with_clssses(
            *self.config['paths']['images']['transformed_test'].
            get(str).split('/'))

    def get_validation_transformed_dirs(self):
        return self._join_with_clssses(
            *self.config['paths']['images']['transformed_validation'].
            get(str).split('/'))

    def _join_with_clssses(self, *dirs):
        return [os.path.join(*dirs, class_name)
                for class_name in self.get_label_names()]

    # data paths
    def get_last_predictions_file(self):
        return self.config['paths']['data']['last_predictions'].get(str)

    def get_train_results_file(self):
        return self.config['paths']['data']['train_results'].get(str)

    def get_test_results_file(self):
        return self.config['paths']['data']['test_results'].get(str)

    def get_predictions_file(self):
        return self.config['paths']['data']['predictions'].get(str)

    # train
    def get_batch_size(self):
        return self.config['train']['batch_size'].get(int)

    def get_epochs(self):
        return self.config['train']['epochs'].get(int)

    def get_n_predictions(self):
        return self.config['train']['predictions'].get(int)
