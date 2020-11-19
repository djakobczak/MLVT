import time
import shutil

import numpy as np
import torch
import torch.nn as nn
import torchvision
from tqdm import tqdm

import logging
from mlvt.server.file_utils import append_to_train_file, \
    create_subdirs_if_not_exist


def init_and_save(path):
    model = Model()
    model.save(path)


LOG = logging.getLogger('MLVT')


class Model:
    def __init__(self, state=None, training_model_path='training_model.pt',
                 best_model_path='best_model.pt', n_out=2, lr=1e-3,
                 gamma=0.5, overwrite=False):
        LOG.info('Model initialization starts...')
        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu")
        self.n_out = n_out
        self.training_model_path = training_model_path
        self.best_model_path = best_model_path
        LOG.info(f'Device: {self.device}')

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
            self.optimizer, milestones=[20, 40, 80], gamma=gamma)

        LOG.info('Model initialized')
        if state:
            self.model_conv.load_state_dict(state['model'])
            self.optimizer.load_state_dict(state['optimizer'])
            LOG.info('Model state loaded')
        if overwrite:
            create_subdirs_if_not_exist(self.training_model_path)
            create_subdirs_if_not_exist(self.best_model_path)
            self.save_states(0, True, 0)

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
              results_save_to=None, n_images=0):
        best_acc = self.get_validation_best_acc()
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
            is_best = vacc > best_acc
            if is_best:
                best_acc = vacc
            self.save_states(vacc, is_best, epoch)

            if results_save_to:
                append_to_train_file(
                    results_save_to,
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

    def save_states(self, acc, is_best, epoch):
        state = {
            'model': self.model_conv.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'acc': acc,
            'epoch': epoch
        }
        torch.save(state, self.training_model_path)
        if is_best:
            shutil.copyfile(self.training_model_path, self.best_model_path)

    def restore_best(self):
        best_state = self.load(self.best_model_path)
        torch.save(best_state, self.training_model_path)

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

    def get_validation_best_acc(self):
        return torch.load(self.best_model_path)['acc']

    def get_validation_current_acc(self):
        return torch.load(self.training_model_path)['acc']
