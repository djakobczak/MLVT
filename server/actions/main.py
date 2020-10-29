from enum import Enum


class Action(Enum):
    TRAIN = 'train'
    TEST = 'test'
    PREDICTION = 'predictions'


class ActionStatus(Enum):
    ONGOING = 'ongoing'
    SUCCESS = 'success'
    FAILED = 'failed'
