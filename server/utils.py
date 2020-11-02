from enum import Enum


class DatasetType(Enum):
    UNLABELLED = 'unlabelled'
    TRAIN = 'train'
    TEST = 'test'
    VALIDATION = 'validation'
