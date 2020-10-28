import torch
import time
from torch.utils.data import Dataset, DataLoader
from dral.logger import LOG
import os
import cv2
from tqdm import tqdm
from torchvision import transforms
from PIL import Image

import torch.nn as nn

import numpy as np
import torchvision

from server.file_utils import load_json


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


def create_csv_file(target_file, data_dir, skips=None, with_labels=True):
    if skips is None:
        skips = []

    with open(target_file, 'w+') as tf:
        label = 0
        for fdir in sorted(os.listdir(data_dir)):
            if fdir in skips:
                continue

            dirpath = os.path.join(data_dir, fdir)
            if os.path.isdir(dirpath):
                LOG.info(f'Start loading from {dirpath}...')
                for f in tqdm(os.listdir(dirpath)):
                    feature_rpath = os.path.join(fdir, f)
                    tf.write(f'{feature_rpath},{label}\n')
                label += 1
    LOG.info(f'CSV file {target_file} saved')


def create_csv_file_without_label(target_file, data_dir, labels=None,
                                  relative_to=None):
    with open(target_file, 'w+') as tf:
        if os.path.isdir(data_dir):
            LOG.info(f'Start loading from {data_dir}...')
            for f in tqdm(os.listdir(data_dir)):
                feature_rpath = os.path.join(data_dir, f)
                try:
                    idx = feature_rpath.index(relative_to)
                    feature_rpath = feature_rpath[idx:]
                except (ValueError, TypeError):
                    pass
                tf.write(f'{feature_rpath}\n')


class LabelledDataset(Dataset):

    def __init__(self, path, transforms=None):
        self.path = path
        self.load()
        self.n_class1 = len(self.all_annotations)
        self.transforms = transforms
        self.n_labels = len(self.annotations)
        self.load_time = 0
        self.trans_time = 0

    def __len__(self):
        return len(self.all_annotations)

    def __getitem__(self, idx):
        start_read = time.time()
        img_path = self.all_annotations[idx]
        img = Image.open(img_path).convert('RGB')
        load_time = time.time() - start_read
        self.load_time += load_time

        target_label = self._get_label(img_path)

        if self.transforms:
            img = self.transforms(img)

        # print(img_path, target_label)
        return img, target_label

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
        self.path = path  # should be an list
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
        img = Image.open(img_path)
        self.load_time += time.time() - start_read

        start_transoform = time.time()
        if self.transforms:
            img = self.transforms(img)
        transofrm_time = time.time() - start_transoform
        self.trans_time += transofrm_time

        return img, img_path

    def load(self):
        self.annotations = \
            load_json(self.path, parse_keys_to=int)[self.unl_label]


def train(csv_file, root_dir):
    if torch.cuda.is_available():
        print("Running on the GPU")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    preprocessed = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    td = LabelledDataset(csv_file, root_dir, transforms=preprocessed)
    dataloader = DataLoader(td, batch_size=128,
                            shuffle=True, num_workers=0)
    return
    model_conv = torchvision.models.resnet18(pretrained=True)
    num_ftrs = model_conv.fc.in_features
    model_conv.fc = nn.Linear(num_ftrs, 2)

    model_conv = model_conv.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model_conv.parameters(), lr=0.002, amsgrad=True)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[500,1000,1500], gamma=0.5)
    epochs = 3
    itr = 1
    p_itr = 200
    model_conv.train()
    total_loss = 0
    loss_list = []
    acc_list = []
    for epoch in tqdm(range(epochs)):
        for samples, labels in tqdm(dataloader):
            samples, labels = samples.to(device), labels.to(device)
            optimizer.zero_grad()
            output = model_conv(samples)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            scheduler.step()

            if itr % p_itr == 0:
                pred = torch.argmax(output, dim=1)
                correct = pred.eq(labels)
                acc = torch.mean(correct.float())
                print('[Epoch {}/{}] Iteration {} -> Train Loss: {:.4f}, Accuracy: {:.3f}'.format(epoch+1, epochs, itr, total_loss/p_itr, acc))
                loss_list.append(total_loss/p_itr)
                acc_list.append(acc)
                total_loss = 0

            itr += 1

    torch.save(model_conv, 'entire_model.pt')
    torch.save(model_conv.state_dict(), 'model_states.pt')


def evaluate(csv_file, root_dir):
    model = torch.load('entire_model.pt')

    preprocessed = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    td = LabelledDataset(csv_file, root_dir, transforms=preprocessed)

    testloader = DataLoader(td, batch_size=128,
                            shuffle=True, num_workers=0)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    correct = 0
    total = 0
    with torch.no_grad():
        for samples, labels in tqdm(testloader):
            samples, labels = samples.to(device), labels.to(device)
            net_out = model(samples)
            for label, net_out in zip(labels, net_out):
                predicted = torch.argmax(net_out)
                if label == predicted:
                    correct += 1
                total += 1
            print("Accuracy: ", round(correct/total, 3))
            return
    print("Accuracy: ", round(correct/total, 3))


def test_loader(csv_file, root_dir):
    preprocessed = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    td = LabelledDataset(csv_file, root_dir, transforms=preprocessed)
    dataloader = DataLoader(td, batch_size=128,
                            shuffle=True, num_workers=0)

    k = 0
    iterator = iter(dataloader)
    while True:
        samples, labels = next(iterator)
        print(len(samples), labels)
        k += len(samples)
        print(f'Total samples: {k}')
        if k > 8000:
            input()


if __name__ == '__main__':
    # root_train_dir = os.path.join('data', 'PetImages', 'Train')
    # csv_train_file = os.path.join('data', 'train_annotations.csv')
    # create_csv_file(csv_file, root_dir,
    #                 skips=['Unknown'])
    # root_test_dir = os.path.join('data', 'PetImages', 'Test')
    # csv_test_file = os.path.join('data', 'test_annotations.csv')
    # create_csv_file(csv_train_file, root_train_dir,
    #                 skips=['Unknown'])
    # annotations_path = os.path.join('server', 'static', 'images',
    #         'server_test', 'unl_annotations.csv')
    # root_dir = os.path.join('server', 'static', 'images', 'TestImages')
    # create_csv_file_without_label(annotations_path, root_dir)
    test = os.path.join('data', 'PetImages', 'Test', 'Cat')
    remove_corrupted_images(test)
    # train(csv_train_file, root_train_dir)
    # evaluate(csv_test_file, root_test_dir)
    # test_loader(csv_test_file, root_test_dir)

    # remove_corrupted_images(os.path.join(root_dir, 'Dog'))
    # remove_corrupted_images(os.path.join(root_dir, 'Cat'))
   
    # itr = iter(dataloader)
    # x, y = next(itr)

    # n_output = 2
    # net = ConvNet(n_output, final_img_size)
    # optimizer = optim.Adam(net.parameters(), lr=1e-3)
    # loss_function = nn.MSELoss()

    # EPOCHS = 10

    # print('Start training')
    # for epoch in range(EPOCHS):
    #     for batch_x, batch_y in tqdm(dataloader):

    #         net.zero_grad()

    #         out = net(batch_x).double()
    #         loss = loss_function(out, batch_y)
    #         loss.backward()
    #         optimizer.step()

    #     print(f'Epoch: {epoch}. Loss: {loss}')

    # correct = 0
    # total = 0
