import logging
import time
import os

import cv2
import torch
from torch.utils.data import Dataset
from PIL import Image

from mlvt.server.file_utils import load_json


LOG = logging.getLogger('MLVT')


def remove_corrupted_images(path):
    removed_files = []
    for f in os.listdir(path):
        full_path = os.path.join(path, f)
        try:
            img = cv2.imread(full_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            if img is None:
                raise OSError
        except OSError:
            os.remove(full_path)
            removed_files.append(full_path)
    LOG.info(f'Removed files: {removed_files}')


class LabelledDataset(Dataset):

    def __init__(self, path, transforms=None, return_paths=False):
        self.path = path
        self.load()
        self.transforms = transforms
        self.n_labels = len(self.annotations)
        self.load_time = 0
        self.trans_time = 0
        self.return_paths = return_paths

    def __len__(self):
        return len(self.all_annotations)

    def __getitem__(self, idx):
        """Access data at specified from dataset.

        Args:
            idx (int): data index

        Returns:
            tuple: Contains loaded image and corresponding label with optionally added image path if class
            has defined return_paths parameter
        """
        start_read = time.time()
        img_path = self.all_annotations[idx]
        img = Image.open(img_path).convert('RGB')
        load_time = time.time() - start_read
        self.load_time += load_time

        target_label = self._get_label(img_path)

        if self.transforms:
            img = self.transforms(img)

        return (img, target_label, img_path) if self.return_paths \
            else (img, target_label)

    def load(self):
        self.annotations = load_json(self.path, parse_keys_to=int)
        self.all_annotations = []
        for label, paths in self.annotations.items():
            self.all_annotations.extend(paths)

    def _get_label(self, img_path):
        for label, paths in self.annotations.items():
            if img_path in paths:
                return torch.tensor(label)


class UnlabelledDataset(Dataset):

    def __init__(self, path, transforms=None, unl_label=255):
        self.path = path  # should be a list
        self.unl_label = unl_label
        self.load()
        self.transforms = transforms
        self.load_time = 0
        self.trans_time = 0

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        start_read = time.time()
        img_path = self.annotations[idx]
        img = Image.open(img_path).convert('RGB')
        self.load_time += time.time() - start_read

        start_transoform = time.time()
        if self.transforms:
            img = self.transforms(img)
        transofrm_time = time.time() - start_transoform
        self.trans_time += transofrm_time
        return img, img_path

    def load(self):
        self.load_time = 0
        self.trans_time = 0
        self.annotations = \
            load_json(self.path, parse_keys_to=int)[self.unl_label]
