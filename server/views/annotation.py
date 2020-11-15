import os

import PIL
import torch
from tqdm import tqdm

from dral.utils import get_resnet_test_transforms
from server.views.base import BaseView
from server.utils import DatasetType
from server.file_utils import update_annotation_file, purge_json_file, \
    load_json


TOTAL_ITEMS = 'Total number of images'


class AnnotationsView(BaseView):

    def search(self):
        unl_json = load_json(self.cm.get_unl_annotations_path(),
                             parse_keys_to=int)
        train_json = load_json(self.cm.get_train_annotations_path(),
                               parse_keys_to=int)
        test_json = load_json(self.cm.get_test_annotations_path(),
                              parse_keys_to=int)
        validation_json = load_json(self.cm.get_validation_annotations_path(),
                                    parse_keys_to=int)
        summary = dict()
        summary['Unlabelled'] = self.pretty_summary(
            self.json_summary(unl_json))
        summary['Train'] = self.pretty_summary(
            self.json_summary(train_json))
        summary['Test'] = self.pretty_summary(
            self.json_summary(test_json))
        summary['Validation'] = self.pretty_summary(
            self.json_summary(validation_json))
        return summary, 200

    def json_summary(self, json_data):
        label_to_cls = self._numeric_to_classname()
        label_to_cls[self.cm.get_unknown_label()] = 'Unlabelled'
        summary = {label_to_cls.get(key): len(val)
                   for key, val in json_data.items()}
        summary[TOTAL_ITEMS] = sum(val for _, val in summary.items())
        return summary

    def pretty_summary(self, summary):
        txt_summary = str()
        for key, val in summary.items():
            txt_summary += f"{key}: <b>{val}</b> images<br>"
        return txt_summary

    def put(self, force, dataset_type, prune):
        return self._handle_annotation_type(dataset_type, force, prune)

    def _handle_annotation_type(self, annotation, force, prune):
        if annotation == 'all':
            self._create_unl_annotation(prune)
            self._create_validation_annotaion(prune)
            self._create_train_annotaion(prune)
            self._create_test_annotation(prune)

        elif annotation == DatasetType.UNLABELLED.value:
            self._create_unl_annotation(prune)

        elif annotation == DatasetType.TRAIN.value:
            self._create_train_annotaion(prune)

        elif annotation == DatasetType.TEST.value:
            self._create_test_annotation(prune)

        elif annotation == DatasetType.VALIDATION.value:
            self._create_validation_annotaion(prune)

        return 'Annotation file has been created', 200

    # !TODO could be written better
    def _create_unl_annotation(self, prune):
        data_dir = self.cm.get_unl_transformed_dir()
        annotation_path = self.cm.get_unl_annotations_path()
        if prune:
            purge_json_file(annotation_path)
        update_annotation_file(annotation_path,
                               data_dir,
                               self.cm.get_unknown_label())

    def _create_train_annotaion(self, prune):
        train_dirs = self.cm.get_train_transformed_dirs()
        labels = self.cm.get_numeric_labels()
        annotation_path = self.cm.get_train_annotations_path()
        if prune:
            purge_json_file(annotation_path)
        for train_dir, label in zip(train_dirs, labels):
            update_annotation_file(annotation_path,
                                   train_dir,
                                   label)

    def _create_validation_annotaion(self, prune):
        valid_dirs = self.cm.get_validation_transformed_dirs()
        labels = self.cm.get_numeric_labels()
        annotation_path = self.cm.get_validation_annotations_path()
        if prune:
            purge_json_file(annotation_path)
        for valid_dir, label in zip(valid_dirs, labels):
            update_annotation_file(annotation_path,
                                   valid_dir,
                                   label)

    def _create_test_annotation(self, prune):
        test_dirs = self.cm.get_test_transformed_dirs()
        labels = self.cm.get_numeric_labels()  # labels hould correspond to dirs
        annotation_path = self.cm.get_test_annotations_path()
        if prune:
            purge_json_file(annotation_path)
        for test_dir, label in zip(test_dirs, labels):
            update_annotation_file(annotation_path,
                                   test_dir,
                                   label)

    def _create_validation_tensor(self):
        validation_dirs = self.cm.get_validation_transformed_dirs()
        labels = self.cm.get_numeric_labels()  # labels should correspond to dirs
        x = torch.empty(size=(0, 3, 224, 224))
        y = torch.empty(size=(0,))
        transforms = get_resnet_test_transforms()
        for validation_dir, label in zip(validation_dirs, labels):
            label = torch.Tensor([label])
            for f in tqdm(os.listdir(validation_dir)):
                path = os.path.join(validation_dir, f)
                img = PIL.Image.open(path)
                img = torch.unsqueeze(transforms(img), 0)

                x = torch.cat((x, img), 0)
                y = torch.cat((y, label), 0)
        torch.save(x, './data/x.pt')
        torch.save(y, './data/y.pt')
