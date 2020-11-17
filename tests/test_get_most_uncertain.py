import numpy as np

from mlvt.model.models import Model
from unittest.mock import patch


@patch('dral.models.Model.__init__', return_value=None)
def test_get_most_uncertain(_):
    model = Model()
    model.n_out = 2
    predictions = np.array([
        [0.51, 0.49],
        [0.8, 0.2],
        [0.6, 0.4],
        [0.11, 0.89],
        [0.75, 0.25],
        [0.99, 0.01],
        [0.1, 0.9],
        [0.45, 0.55]
    ])
    paths = np.array([
        'path/to/img1',
        'path/to/img2',
        'path/to/img3',
        'path/to/img4',
        'path/to/img5',
        'path/to/img6',
        'path/to/img7',
        'path/to/img8',
    ])
    expected = {
        0: ['path/to/img1', 'path/to/img3'],
        1: ['path/to/img8', 'path/to/img4']
    }
    mapping = model._get_most_uncertain(predictions, paths, 2)
    assert mapping == expected
