from enum import Enum
from multiprocessing import Value


class DatasetType(Enum):
    UNLABELLED = 'unlabelled'
    TRAIN = 'train'
    TEST = 'test'
    VALIDATION = 'validation'


test_image_counter = Value('i', 0)
