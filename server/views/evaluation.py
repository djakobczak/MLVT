from torch.utils.data import DataLoader

from dral.datasets import LabelledDataset
from dral.utils import get_resnet18_default_transforms
from server.views.base import MLView


class EvaluateView(MLView):
    def search(self):
        model = self.load_model()
        test_ds = LabelledDataset(
            self.cm.get_test_annotations_path(),
            self.cm.get_n_labels(),
            get_resnet18_default_transforms())
        testlaoder = DataLoader(
            test_ds, batch_size=self.cm.get_batch_size(),
            shuffle=True, num_workers=2)
        acc = model.evaluate(testlaoder)
        print(f'Time: load {test_ds.load_time}, '
              f'transofrm: {test_ds.trans_time}')
        return acc, 200
