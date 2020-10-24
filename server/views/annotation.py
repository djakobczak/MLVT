from server.views.base import BaseView
from server.utils import DatasetType
from server.file_utils import update_annotation_file, prune_annotation_file


class AnnotationsView(BaseView):
    def put(self, force, dataset_type, prune):
        return self._handle_annotation_type(dataset_type, force, prune)

    def _handle_annotation_type(self, annotation, force, prune):
        if annotation == 'all':
            self._create_unl_annotation(prune)
            self._create_train_annotaion(prune)
            self._create_test_annotation(prune)

        elif annotation == DatasetType.UNLABELLED.value:
            self._create_unl_annotation(prune)

        elif annotation == DatasetType.TRAIN.value:
            self._create_train_annotaion(prune)

        elif annotation == DatasetType.TEST.value:
            self._create_test_annotation(prune)

        return 'Annotation file has been created', 200

    # !TODO could written better
    def _create_unl_annotation(self, prune):
        data_dir = self.cm.get_unl_transformed_dir()
        annotation_path = self.cm.get_unl_annotations_path()
        if prune:
            prune_annotation_file(annotation_path)
        update_annotation_file(annotation_path,
                               data_dir,
                               self.cm.get_unknown_label())

    def _create_train_annotaion(self, prune):
        train_dirs = self.cm.get_train_transformed_dirs()
        labels = self.cm.get_numeric_labels()
        annotation_path = self.cm.get_train_annotations_path()
        if prune:
            prune_annotation_file(annotation_path)
        for train_dir, label in zip(train_dirs, labels):
            update_annotation_file(annotation_path,
                                   train_dir,
                                   label)

    def _create_test_annotation(self, prune):
        test_dirs = self.cm.get_test_transformed_dirs()
        labels = self.cm.get_numeric_labels()  # labels hould correspond to dirs
        annotation_path = self.cm.get_test_annotations_path()
        if prune:
            prune_annotation_file(annotation_path)
        for test_dir, label in zip(test_dirs, labels):
            update_annotation_file(annotation_path,
                                   test_dir,
                                   label)
