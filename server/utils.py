from ctypes import c_wchar_p
from enum import Enum
from multiprocessing import Value


class DatasetType(Enum):
    UNLABELLED = 'unlabelled'
    TRAIN = 'train'
    TEST = 'test'
    VALIDATION = 'validation'


test_image_counter = Value('i', 0)
last_checksum = Value(c_wchar_p, '')
