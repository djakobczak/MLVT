import sys
import numpy as np
from numpy.random import default_rng
import time

import gym
from gym import spaces
import os
from torchvision import transforms

from stable_baselines3 import DQN
from stable_baselines3.dqn import CnnPolicy
from stable_baselines3.common.env_checker import check_env
import gym
import torch as th
import torch.nn as nn
import torchvision

from stable_baselines3 import PPO, DQN
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from custom_dataset import TrainDataset, UnlabelledDataset, create_csv_file_without_label
from torch.utils.data import DataLoader


class CustomCNN(CnnPolicy):
    """
    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """

    def __init__(self, observation_space: gym.spaces.Box, action_space: gym.spaces.Box):
        super(CustomCNN, self).__init__(observation_space, features_dim)
        # We assume CxHxW images (channels first)
        # Re-ordering will be done by pre-preprocessing or wrapper
        self.cnn = torchvision.models.resnet18(pretrained=True)
        num_ftrs = self.cnn.fc.in_features
        print(num_ftrs)
        # self.cnn.fc = nn.Linear(num_ftrs, 2)
        self.device = th.device("cuda:0" if th.cuda.is_available() else "cpu")
        self.cnn = self.cnn.to(self.device)
        with th.no_grad():
            n_flatten = self.cnn(
                th.as_tensor(observation_space.sample()[None]).float()
            ).shape[1]
        self.cnn.fc = nn.Sequential(nn.Linear(n_flatten, num_ftrs), nn.ReLU())

    def forward(self, observations: th.Tensor) -> th.Tensor:
        observations = th.Tensor(observations)
        observations = observations.to(self.device)
        return self.cnn(observations)


class ClassificationEnv(gym.Env):

    def __init__(self, dataloader, img_size, batch_size):
        super(ClassificationEnv, self).__init__()
        self.dataloader = dataloader
        self.data_iterator = iter(dataloader)
        self.img_size = img_size
        self.correct_shape = (batch_size, 3, self.img_size, self.img_size)
        self._counter = 0
        self.stats = {'good': 0, 'wrong': 0}

        self.action_space = spaces.Box(
            shape=(self.correct_shape[0],),
            low=0, high=1, dtype=np.int8)

        self.observation_space = spaces.Box(
            shape=self.correct_shape,
            low=-3, high=3, dtype=np.float32)

    def reset(self):
        self._counter = 0
        self.stats = {'good': 0, 'wrong': 0}
        self._state, self._true_label = next(self.data_iterator)
        self._state = self._state.numpy()
        return self._state

    def step(self, actions):
        reward = 0

        # predicted_label = action

        # print(f'Predicted label: {actions},'
        #       f'true_label: {self._true_label}')

        for predicted_label, true_label in zip(actions, self._true_label):
            if predicted_label == true_label:
                self.stats['good'] += 1
                reward += 1
            else:
                self.stats['wrong'] += 1

        self._counter += 1
        info = {}
        try:
            self._state, self._true_label = next(self.data_iterator)
            self._state = self._state.numpy()
            done = False
        except StopIteration:
            self._state = np.zeros(self.correct_shape,
                                   dtype=np.float32)
            done = True
            info = self.stats
            print(f'End of epoch, results: {self.stats}')

        return self._state, reward, done, info

    def enable_evaluating(self, en):  # !TODO context manager
        self.evaluating = en
        self.storage = self.dm.test if en else self.dm.train

    def get_counter(self):
        return self._counter


if __name__ == "__main__":
    root_train_dir = os.path.join('data', 'PetImages', 'Train')
    csv_train_file = os.path.join('data', 'train_annotations.csv')
    root_test_dir = os.path.join('data', 'PetImages', 'Test')
    csv_test_file = os.path.join('data', 'test_annotations.csv')
    root_unl_dir = os.path.join('data', 'PetImages', 'Unlabelled')
    csv_unl_file = os.path.join('data', 'unl_annotations.csv')
    create_csv_file_without_label(csv_unl_file, root_unl_dir)
    preprocessed = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    BATCH_SIZE = 32
    td = TrainDataset(csv_test_file, root_test_dir, transforms=preprocessed)
    dataloader = DataLoader(td, batch_size=BATCH_SIZE,
                            shuffle=True, num_workers=0)

    ud = UnlabelledDataset(csv_unl_file, root_unl_dir, transforms=preprocessed)
    unl_dataloader = DataLoader(ud, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    iterator = iter(dataloader)

    env = ClassificationEnv(dataloader, 224, BATCH_SIZE)

    policy_kwargs = dict(
        features_extractor_class=CustomCNN,
        features_extractor_kwargs=dict(features_dim=512)
    )
    model = DQN(CustomCNN, env, verbose=1, buffer_size=1000,  batch_size=BATCH_SIZE, learning_starts=5000)
    model.learn(10000)
