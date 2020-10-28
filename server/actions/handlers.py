from torch.utils.data import DataLoader

from server.actions.base import MLAction
from server.actions.main import ActionStatus
from server.file_utils import append_to_json_file, \
    get_last_n_images_key
from dral.datasets import LabelledDataset
from dral.logger import LOG
from dral.utils import get_resnet18_default_transforms


# !TODO decorator
def train():
    try:
        ml_action = MLAction()
        model = ml_action.load_model()

        losses, accs = model.train(
                ml_action.get_train_loader(),
                ml_action.cm.get_epochs())
        LOG.info(f'losses: {losses}, accs: {accs}')
        ml_action.save_model(model)

        # save to output to file
        append_to_json_file(
            ml_action.cm.get_train_results_file(),
            {'loss': losses,
             'acc': accs,
             'n_images': len(ml_action.train_dataset)})
    except Exception:
        return ActionStatus.FAILED
    return ActionStatus.SUCCESS


def test():
    ml_action = MLAction()
    model = ml_action.load_model()
    test_ds = LabelledDataset(
        ml_action.cm.get_test_annotations_path(),
        get_resnet18_default_transforms())
    testlaoder = DataLoader(
        test_ds, batch_size=ml_action.cm.get_batch_size(),
        shuffle=True, num_workers=2)

    LOG.info('Start testing model')
    acc = model.evaluate(testlaoder)
    LOG.info(f'Model accuraccy: {acc}')

    append_to_json_file(
        ml_action.cm.get_test_results_file(),
        {'acc': acc,
         'n_images': get_last_n_images_key(
             ml_action.cm.get_train_results_file())
         })
    return ActionStatus.SUCCESS
