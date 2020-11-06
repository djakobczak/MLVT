from torch.utils.data import DataLoader

from server.actions.base import MLAction
from server.actions.main import ActionStatus
from server.exceptions import AnnotationException
from server.file_utils import append_to_json_file, \
    get_last_n_images_key
from dral.datasets import LabelledDataset
from dral.logger import LOG
from dral.utils import get_resnet_test_transforms
from server.predictions_manager import PredictionsManager


# !TODO decorator for saving?
def train(**kwargs):
    try:
        ml_action = MLAction()
        model = ml_action.load_model()
        epochs = kwargs.get('epochs') or ml_action.cm.get_epochs()
        batch_size = kwargs.get('batch_size') or ml_action.cm.get_batch_size()
        accs, losses, val_accs, val_losses = model.train(
                ml_action.get_train_loader(batch_size),
                epochs,
                ml_action.get_validation_loader(batch_size))
        LOG.info(f'losses: {losses}, accs: {accs}')
        ml_action.save_model(model)

        # save to output to file
        append_to_json_file(
            ml_action.cm.get_train_results_file(),
            {'losses': losses,
             'accs': accs,
             'val_accs': val_accs,
             'val_losses': val_losses,
             'n_images': len(ml_action.train_dataset)})
    except AnnotationException as e:
        LOG.error(f'Training failed: {e}')
        return ActionStatus.FAILED, 'Please, annote some images'
    except ValueError as e:
        LOG.error(f'Training failed: {e}')
        return ActionStatus.FAILED, 'Unknwon error occured durning training'
    return ActionStatus.SUCCESS, 'Training completed'


def test(**kwargs):
    ml_action = MLAction()
    model = ml_action.load_model()
    test_ds = LabelledDataset(
        ml_action.cm.get_test_annotations_path(),
        get_resnet_test_transforms())
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
    return ActionStatus.SUCCESS, 'Test completed'


def predict(**kwargs):
    n_predictions = kwargs.get('n_predictions')
    random = kwargs.get('random')
    balance = kwargs.get('balance')
    ml_action = MLAction()
    model = ml_action.load_model()
    unl_loader = ml_action.get_unl_loader()

    LOG.info('Ask model to predict images with parameters: '
             f'random={random}, balance={balance}')

    pm = PredictionsManager(ml_action.cm.get_n_labels(),
                            ml_action.cm.get_predictions_file())
    pm.get_new_predictions(
        n_predictions, model=model,
        dataloader=unl_loader, random=random, balance=balance)
    return ActionStatus.SUCCESS, 'Prediction completed'