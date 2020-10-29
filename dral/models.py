import os
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms
from tqdm import tqdm

from dral.datasets import LabelledDataset
from dral.logger import LOG
from server.file_utils import save_json, load_json


def init_and_save(path):
    model = Model()
    model.save(path)


class Model:
    def __init__(self, model=None, n_out=2):
        LOG.info('Model initialization starts...')
        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu")
        self.n_out = n_out
        LOG.info(f'CUDA: {self.device}')
        if model is None:
            model = torchvision.models.resnet34(pretrained=True)
            # freeze conv layers
            for param in model.parameters():
                param.requires_grad = False
            num_ftrs = model.fc.in_features
            model.fc = nn.Sequential(
                nn.Linear(num_ftrs, self.n_out),
                nn.Softmax(dim=0))
            LOG.info('New model created')
        self.model_conv = model.to(self.device)

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(
            self.model_conv.parameters(), lr=0.0005, amsgrad=True)
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(
            self.optimizer, milestones=[100, 200, 300], gamma=0.5)

    def __call__(self, batch_x):
        return self.model_conv(batch_x)

    def predict(self, batch_x):
        self.model_conv.eval()
        with torch.no_grad():
            batch_x = batch_x.to(self.device)
            return self(batch_x)

    def predict_all(self, dataloader):
        self.model_conv.eval()
        transforms_time = 0
        feedforward_time = 0
        predictions = torch.empty((0, 2), device=self.device)
        paths = []
        with torch.no_grad():
            for batch_x, img_paths in tqdm(dataloader):
                batch_x = batch_x.to(self.device)

                start_pred = time.time()
                prediction = self(batch_x)
                feedforward_time += time.time() - start_pred
                start_transoform = time.time()
                predictions = torch.cat((predictions, prediction), 0)
                transforms_time += time.time() - start_transoform
                for path in img_paths:
                    paths.append(path)
        print(f"[DEBUG] loading time: {transforms_time}, feedforward time: {feedforward_time}")

        return predictions.cpu().numpy(), np.array(paths)

    def get_predictions(self, dataloader, n_predictions, predictions_path,
                        random, balance=True, from_file=False):
        if from_file:
            LOG.info('Load predictions from file.')
            predictions, paths = self._load_predictions(predictions_path)
        else:
            LOG.info('Predict images.')
            predictions, paths = self.predict_all(dataloader)
            self._save_predictions(
                predictions_path, predictions.tolist(), paths.tolist())

        if random:
            return self._get_random(
                predictions, paths, n_predictions, balance)
        else:
            return self._get_most_uncertain(
                predictions, paths, n_predictions, balance)

    def _save_predictions(self, path, predictions, paths):
        save_json(path, {
            'predictions': predictions,
            'paths': paths
        })

    def _load_predictions(self, path):  # !TODO
        json_data = load_json(path)
        return json_data.get('predictions'), json_data.get('paths')

    def _get_most_uncertain(self, predictions, paths, n, balance=True):
        diffs = np.apply_along_axis(lambda x: np.absolute(x[0] - x[1]),
                                    1, predictions)
        labels = np.apply_along_axis(lambda x: np.argmax(x),
                                     1, predictions)
        idxs = np.argsort(diffs, axis=0)
        labels = labels[idxs]
        paths = paths[idxs]

        if balance:
            out_idxs = self._get_balanced_predictions(labels, n)
        else:
            out_idxs = range(2*n)

        out_labels = labels[out_idxs]
        out_paths = paths[out_idxs]
        out_mapping = {label: [] for label in range(self.n_out)}
        for out_label, out_path in zip(out_labels, out_paths):
            out_mapping[out_label].append(out_path)

        return out_mapping

    def _get_random(self, predictions, paths, n, balance=True):
        labels = np.apply_along_axis(lambda x: np.argmax(x),
                                     1, predictions)
        if balance:
            out_idxs = self._get_balanced_predictions(labels, n)
        else:
            out_idxs = range(2*n)
        out_labels = labels[out_idxs]
        out_paths = paths[out_idxs]

        return self._create_mapping(out_labels, out_paths)

    def _get_balanced_predictions(self, labels, n):
        out_idx = []
        ctr = {label: 0 for label in range(self.n_out)}
        for idx, label in enumerate(labels):
            if ctr[label] < n:
                out_idx.append(idx)
            if len(out_idx) >= 2 * n:
                return out_idx
            ctr[label] += 1
        return out_idx

    def _create_mapping(self, labels, paths):
        label_paths_mapping = {label: [] for label in range(self.n_out)}
        for label, path in zip(labels, paths):
            label_paths_mapping[label].append(path)

        return label_paths_mapping

    def train(self, dataloader, epochs):
        self.model_conv.train()  # training mode
        total_loss = 0
        itr = 0
        loss_list = []
        acc_list = []
        for epoch in range(epochs):
            for samples, labels in tqdm(dataloader):
                samples, labels = samples.to(self.device), \
                                  labels.to(self.device)
                self.optimizer.zero_grad()
                output = self.model_conv(samples)
                loss = self.criterion(output, labels)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
                self.scheduler.step()

                itr += 1
            pred = torch.argmax(output, dim=1)
            correct = pred.eq(labels)
            acc = torch.mean(correct.float())
            LOG.info('[Epoch {}/{}] Iteration {} -> Train Loss: {:.4f}, '
                     'Accuracy: {:.3f}'.format(
                        epoch+1, epochs, itr, total_loss/itr, acc))
            loss_list.append(total_loss/itr)
            acc_list.append(float(acc.cpu()))
            total_loss = 0

        return loss_list, acc_list

    def save(self, name):
        torch.save(self.model_conv, name)

    @staticmethod
    def load(path):
        print(f'load from path: {path}')
        return torch.load(path)

    def evaluate(self, testloader):
        correct = 0
        total = 0
        self.model_conv.eval()

        with torch.no_grad():
            epoch_start = time.time()
            epoch_tt = 0
            for samples, labels in tqdm(testloader):
                samples, labels = samples.to(self.device), \
                                  labels.to(self.device)

                tt_start = time.time()
                net_out = self(samples)
                epoch_tt += (time.time() - tt_start)
                for label, net_out in zip(labels, net_out):
                    predicted = torch.argmax(net_out)
                    if label == predicted:
                        correct += 1
                    total += 1
            epoch_res = time.time()-epoch_start
            LOG.info(f'Epoch overral time: {epoch_res},'
                     f' feedforward time: {epoch_tt}')
        return round(correct/total, 3)


BATCH_SIZE = 128
MODEL_NAME = 'model.pt'


def main():
    root_train_dir = os.path.join('data', 'PetImages', 'Train')
    csv_train_file = os.path.join('data', 'train_annotations.csv')
    root_test_dir = os.path.join('data', 'PetImages', 'Test')
    csv_test_file = os.path.join('data', 'test_annotations.csv')

    preprocessed = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    td = LabelledDataset(csv_train_file, root_train_dir, transforms=preprocessed)
    trainloader = DataLoader(td, batch_size=BATCH_SIZE,
                             shuffle=True, num_workers=0)
    model = Model()
    model.train(trainloader, epochs=3)
    model.save(MODEL_NAME)

    td = LabelledDataset(csv_test_file, root_test_dir, transforms=preprocessed)
    testloader = DataLoader(td, batch_size=BATCH_SIZE,
                            shuffle=True, num_workers=0)

    model.evaluate(testloader)


if __name__ == "__main__":
    path1 = os.path.join('data', 'PetImages', 'Unlabelled', '4001.jpg')
    path2 = os.path.join('data', 'PetImages', 'Test', 'Cat', '2.jpg')

    device = torch.device(
        "cuda:0" if torch.cuda.is_available() else "cpu")
    preprocessed = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    model = torch.load(MODEL_NAME)
    predictions = model(img1)
    print(predictions)
    predictions = model(img2)
    print(predictions)

    model_conv = torchvision.models.resnet18(pretrained=True)
    num_ftrs = model_conv.fc.in_features
    model_conv.fc = nn.Sequential(
        nn.Linear(num_ftrs, 256),
        nn.Linear(256, 2))
