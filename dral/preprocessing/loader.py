import os
from pathlib import Path

import cv2
from tqdm import tqdm
import torch
from PIL import Image
from torchvision import transforms

from dral.config.config_manager import ConfigManager
from dral.annotations import create_csv_file, label_samples
from dral.utils import get_resnet18_default_transforms


class DataLoader:
    def __init__(self, cm):
        self.cm = cm
        self.IMG_SIZE = cm.get_img_size()

    def copy_all(self, src, dst):
        Path(dst).mkdir(parents=True, exist_ok=True)
        for f in tqdm(os.listdir(src)):
            src_path = os.path.join(src, f)
            if os.path.isfile(src_path):
                img = cv2.imread(src_path)
                img = self._rescale(img)
                dst_path = os.path.join(dst, f)
                cv2.imwrite(dst_path, img)

    def copy_with_transforms(self, src, dst, transforms):
        Path(dst).mkdir(parents=True, exist_ok=True)
        for f in tqdm(os.listdir(src)):
            src_path = os.path.join(src, f)
            if os.path.isfile(src_path):
                img = Image.open(src_path).convert('RGB')
                img = transforms(img)
                dst_path = os.path.join(dst, f)
                img.save(dst_path)

    def _rescale(self, img):
        """Rescale image to the specified in config file size.
        If rescale_with_crop flag is specified then the image is firstly
        resized that the shorter side has desired size then it is cropped.

        Args:
            img (array): array representation of an image

        Returns:
            array: array representation of rescaled image
        """
        if self.cm.do_rescale_with_crop():
            (h, w) = img.shape[:2]
            if h > w:
                ratio = h / w
                # dims: (w, h)
                dims = (self.IMG_SIZE, int(self.IMG_SIZE*ratio))
                img = cv2.resize(img, dims)
                crop_idx = (dims[1] - self.IMG_SIZE) // 2
                img = img[crop_idx:self.IMG_SIZE+crop_idx, :]
            else:
                ratio = w / h
                dims = (int(self.IMG_SIZE*ratio), self.IMG_SIZE)
                img = cv2.resize(img, dims)
                crop_idx = (dims[0] - self.IMG_SIZE) // 2
                img = img[:, crop_idx:self.IMG_SIZE+crop_idx]
        else:
            img = cv2.resize(img, (self.IMG_SIZE, self.IMG_SIZE))
        return img


if __name__ == "__main__":
    cm = ConfigManager('testset')
    dl = DataLoader(cm)
    transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224)])
    dl.copy_with_transforms(cm.get_raw_images(),
                            cm.get_transformed_dir(),
                            transforms)

    # create_csv_file(cm.get_unl_annotations(),
    #                 cm.get_processed_dir())
    # create_csv_file(cm.get_train_annotations(),
    #                 None, labels='abc')
    # paths = ['.\\server\\static\\testset\\images\\4002.jpg', '.\\server\\static\\testset\\images\\4088.jpg']
    # label_samples(cm.get_unl_annotations(), None, paths, None)
