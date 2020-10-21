import os

from connexion import request

from dral.annotations import create_csv_file
from server.views.base import BaseView
from server.utils import DatasetType


class AnnotationsView(BaseView):
    def put(self):
        annotation = request.args.get('dataset_type')
        force = request.args.get('force')
        return self._handle_annotation_type(annotation, force, "w+")

    def _handle_annotation_type(self, annotation, force, mode):
        csv_dir = self.cm.get_annotation_dir()
        if annotation == DatasetType.UNLABELLED.value:
            data_dir = self.cm.get_unl_transformed_dir()
            name = self.cm.get_unl_annotations_filename()
            annotation_file = os.path.join(csv_dir, name)
            exc = create_csv_file(annotation_file, data_dir,
                                  force=force, header='paths',
                                  mode=mode)

        elif annotation == DatasetType.TRAIN.value:
            name = self.cm.get_train_annotations_filename()
            annotation_file = os.path.join(csv_dir, name)
            exc = create_csv_file(annotation_file, None,
                                  force=force, mode=mode)

        elif annotation == DatasetType.TEST.value:
            test_dirs = self.cm.get_test_transformed_dirs()
            name = self.cm.get_test_annotations_filename()
            labels = self.cm.get_numeric_labels()
            annotation_file = os.path.join(csv_dir, name)

            # clear the file
            with open(annotation_file, "w") as f:
                f.write('paths,label\n')

            for label, test_dir in zip(labels, test_dirs):
                exc = create_csv_file(annotation_file, test_dir,
                                      force=force, header=None,
                                      mode="a+", label=label)
                if exc:
                    break

        return (f'File {annotation_file} has been created', 200) \
            if exc is None else (exc, 400)
