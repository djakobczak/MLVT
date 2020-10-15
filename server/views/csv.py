from enum import Enum
import os

from connexion import request

from dral.annotations import create_csv_file
from server.views.base import BaseView


class Annotation(Enum):
    UNLABELLED = 'unlabelled'
    TRAIN = 'train'
    TEST = 'test'


class CsvView(BaseView):
    def put(self, name):
        annotation = request.args.get('annotation')
        force = request.args.get('force')
        append = request.json.get('append')
        mode = "a+" if append else "w+"
        if annotation:
            return self._handle_annotation_type(annotation, force, mode)

        data_dir = request.json.get('data_dir', self.cm.get_processed_dir())
        csv_dir = request.json.get('csv_dir', self.cm.get_annotation_dir())
        label = request.json.get('label')
        labels_header = request.json.get('labels_header')

        if name == 'default-unl':
            name = self.cm.get_unl_annotations_filename()
        elif name == 'default-train':
            name = self.cm.get_train_annotations_filename()
        else:
            name += '.csv'

        annotation_file = os.path.join(csv_dir, name)
        exc = create_csv_file(annotation_file, data_dir,
                              force=force, labels_header=labels_header,
                              label=label, mode=mode)

        return (f'File {annotation_file} has been created', 200) \
            if exc is None else (exc, 400)

    def _handle_annotation_type(self, annotation, force, mode):
        csv_dir = self.cm.get_annotation_dir()
        if annotation == Annotation.UNLABELLED.value:
            data_dir = self.cm.get_transformed_dir()
            name = self.cm.get_unl_annotations_filename()
            annotation_file = os.path.join(csv_dir, name)
            exc = create_csv_file(annotation_file, data_dir,
                                  force=force, header='paths',
                                  mode=mode)

        elif annotation == Annotation.TRAIN.value:
            name = self.cm.get_train_annotations_filename()
            annotation_file = os.path.join(csv_dir, name)
            exc = create_csv_file(annotation_file, None,
                                  force=force, mode=mode)

        elif annotation == Annotation.TEST.value:
            test_dirs = self.cm.get_test_dirs()
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
