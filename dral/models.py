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
                nn.Softmax(dim=1))
            LOG.info('New model created')
        self.model_conv = model.to(self.device)

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(
            self.model_conv.parameters(), lr=0.0005, amsgrad=True)
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(
            self.optimizer, milestones=[10, 20, 30], gamma=0.5)

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
        LOG.debug(f"[DEBUG] loading time: {transforms_time}, feedforward time: {feedforward_time}")
        return predictions.cpu().numpy(), np.array(paths)

    def train(self, dataloader, epochs, validation_dataloader=None):
        self.model_conv.train()  # training mode
        loss_list = []
        acc_list = []
        validation_acc = []
        for epoch in range(epochs):
            total_loss = 0
            itr = 0
            total_corrects = torch.empty((0,), device=self.device, dtype=int)
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
                total_corrects = torch.cat(
                    (total_corrects, pred.eq(labels)), 0)
            acc = torch.mean(total_corrects.float())
            LOG.info('[Epoch {}/{}] Iteration {} -> Train Loss: {:.4f}, '
                     'Accuracy: {:.3f}'.format(
                        epoch+1, epochs, itr, total_loss/itr, acc))
            loss_list.append(total_loss/itr)
            acc_list.append(float(acc.cpu()))
            total_loss = 0
            if validation_dataloader:
                vacc = self.evaluate(validation_dataloader)
                validation_acc.append(vacc)
                LOG.info(f"Validation accuraccy: {vacc}")

        return loss_list, acc_list, validation_acc

    def save(self, name):
        torch.save(self.model_conv, name)

    @staticmethod
    def load(path):
        return torch.load(path)

    def evaluate(self, testloader):
        self.model_conv.eval()
        total_corrects = torch.empty((0,), device=self.device, dtype=int)

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
                calc_time += (time.time() - calc_start)

            epoch_res = time.time()-epoch_start
            LOG.debug(f'Epoch overral time: {epoch_res},'
                      f' feedforward time: {ff_time},'
                      f' to gpu time: {to_gpu_time},'
                      f' calc time: {calc_time}')
        return float(torch.mean(total_corrects.float()).cpu().numpy())


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
