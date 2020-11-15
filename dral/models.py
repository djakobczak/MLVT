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
from server.file_utils import append_to_train_file


def init_and_save(path):
    model = Model()
    model.save(path)


class Model:
    def __init__(self, model=None, n_out=2, lr=1e-3):
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
                nn.Softmax(dim=1))
            LOG.info('New model created')
        self.model_conv = model.to(self.device)

        self.criterion = nn.CrossEntropyLoss().to(self.device)
        self._init_optimizer(lr)
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(
            self.optimizer, milestones=[20, 40, 80], gamma=0.5)

    def _init_optimizer(self, lr=0.001):
        self._lr = lr
        self.optimizer = torch.optim.Adam(
            self.model_conv.parameters(), lr=lr, amsgrad=True)

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

                paths.extend(img_paths)
        LOG.debug(
            f'[DEBUG] loading time: {transforms_time},'
            f'feedforward time: {feedforward_time}')
        return predictions.cpu().numpy(), np.array(paths)

    def train(self, train_loader, epochs, validation_loader,
              save_to=None, n_images=0):
        # switch to training mode
        self.model_conv.train()

        LOG.info("Training started")
        train_accs = []
        train_losses = []
        val_accs = []
        val_losses = []
        for epoch in range(epochs):
            tacc, tloss = self._train(train_loader)
            LOG.info('[Epoch {}/{}] -> Train Loss: {:.4f}, '
                     'Accuracy: {:.3f}'.format(
                        epoch+1, epochs, tloss, tacc))

            vacc, vloss = self.evaluate(validation_loader)
            if save_to:
                append_to_train_file(
                    save_to,
                    {'train_acc': [tacc],
                     'train_loss': [tloss],
                     'val_acc': [vacc],
                     'val_loss': [vloss],
                     'n_images': [n_images]})

            train_accs.append(tacc)
            train_losses.append(tloss)
            val_accs.append(vacc)
            val_losses.append(vloss)
        return train_accs, train_losses, val_accs, val_losses

    def _train(self, dataloader):
        losses = torch.empty((0,), device=self.device, dtype=float)
        total_corrects = torch.empty((0,), device=self.device, dtype=int)
        for samples, labels in tqdm(dataloader):
            samples, labels = samples.to(self.device), \
                            labels.to(self.device)
            net_out = self.model_conv(samples)
            loss = self.criterion(net_out, labels)
            losses = torch.cat(
                (losses, self.criterion(net_out, labels).reshape(1)), 0)

            # compute gradient and do SGD step
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            self.scheduler.step()

            pred = torch.argmax(net_out, dim=1)
            total_corrects = torch.cat(
                (total_corrects, pred.eq(labels)), 0)
        acc = float(torch.mean(total_corrects.float()).cpu())
        loss = float(torch.mean(losses).cpu())
        return acc, loss

    def save(self, name):
        torch.save(self.model_conv, name)

    @staticmethod
    def load(path):
        return torch.load(path)

    def evaluate(self, testloader):
        # switch to evaluate mode
        self.model_conv.eval()

        total_corrects = torch.empty((0,), device=self.device, dtype=int)
        losses = torch.empty((0,), device=self.device, dtype=float)
        with torch.no_grad():
            epoch_start = time.time()
            ff_time = 0
            to_gpu_time = 0
            calc_time = 0
            for samples, labels in tqdm(testloader):
                to_gpu_start = time.time()
                samples, labels = samples.to(self.device), \
                    labels.to(self.device)
                to_gpu_time += (time.time() - to_gpu_start)

                ff_start = time.time()
                net_out = self(samples)
                ff_time += (time.time() - ff_start)

                calc_start = time.time()
                pred = torch.argmax(net_out, dim=1)
                total_corrects = torch.cat(
                    (total_corrects, pred.eq(labels)), 0)

                # store loss (.item() is very slow operation)
                losses = torch.cat(
                    (losses, self.criterion(net_out, labels).reshape(1)), 0)
                calc_time += (time.time() - calc_start)

            epoch_res = time.time()-epoch_start
            LOG.info(f'Epoch overral time: {epoch_res},'
                     f' feedforward time: {ff_time},'
                     f' to gpu time: {to_gpu_time},'
                     f' calc time: {calc_time}')
            acc = float(torch.mean(total_corrects.float()).cpu().numpy())
            loss = float(torch.mean(losses).cpu())
            LOG.info(f"Evaluation stats: acc: {acc}, loss: {loss}")
        return acc, loss

    def test(self, testloader):
        # switch to evaluate mode
        self.model_conv.eval()

        total_corrects = torch.empty((0,), device=self.device, dtype=int)
        predictions = torch.empty((0, 2), device=self.device, dtype=float)
        paths = []
        losses = torch.empty((0,), device=self.device, dtype=float)
        with torch.no_grad():
            for samples, labels, img_paths in tqdm(testloader):
                samples, labels = samples.to(self.device), \
                    labels.to(self.device)

                net_out = self(samples)
                predictions = torch.cat((predictions, net_out), 0)

                predicted_labels = torch.argmax(net_out, dim=1)
                total_corrects = torch.cat(
                    (total_corrects, predicted_labels.eq(labels)), 0)

                # store loss (.item() is very slow operation)
                losses = torch.cat(
                    (losses, self.criterion(net_out, labels).reshape(1)), 0)
                paths.extend(img_paths)

            acc = float(torch.mean(total_corrects.float()).cpu().numpy())
            loss = float(torch.mean(losses).cpu())
            LOG.info(f"Evaluation stats: acc: {acc}, loss: {loss}")
        return acc, loss, predictions.cpu().numpy(), np.array(paths)

    def get_lr(self):
        return self._lr


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
    predictions = model(img2)

    model_conv = torchvision.models.resnet18(pretrained=True)
    num_ftrs = model_conv.fc.in_features
    model_conv.fc = nn.Sequential(
        nn.Linear(num_ftrs, 256),
        nn.Linear(256, 2))
