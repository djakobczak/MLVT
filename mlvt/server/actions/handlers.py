import logging
import torch
from torch.utils.data import DataLoader

from mlvt.server.actions.base import MLAction
from mlvt.server.actions.main import ActionStatus
from mlvt.server.exceptions import AnnotationException
from mlvt.server.file_utils import append_to_json_file
from mlvt.model.datasets import LabelledDataset
from mlvt.model.utils import get_resnet_test_transforms
from mlvt.server.predictions_manager import PredictionsManager
from mlvt.server.utils import test_image_counter


LOG = logging.getLogger('MLVT')


def train(**kwargs):
    try:
        ml_action = MLAction()
        model = ml_action.load_training_model()
        epochs = kwargs.get('epochs') or ml_action.cm.get_epochs()
        batch_size = kwargs.get('batch_size') or ml_action.cm.get_batch_size()
        accs, losses, val_accs, val_losses = model.train(
                ml_action.get_train_loader(batch_size),
                epochs,
                ml_action.get_validation_loader(batch_size),
                results_save_to=ml_action.cm.get_train_results_file(),
                n_images=len(ml_action.train_dataset))
        LOG.info(f'losses: {losses}, accs: {accs}')
        # ml_action.save_training_model(model)

        torch.cuda.empty_cache()
        return ActionStatus.SUCCESS, 'Training completed'
    except AnnotationException as e:
        LOG.error(f'Training failed: {e}')
        return ActionStatus.FAILED, 'Please, annotate some images'
    # except Exception as e:
    #     LOG.error(f'Training failed: {e}')
    #     return ActionStatus.FAILED, \
    #         f'Unknwon error occured durning training ({str(e)})'
    finally:
        torch.cuda.empty_cache()


def test(**kwargs):
    try:
        ml_action = MLAction()
        model = ml_action.load_training_model()
        test_ds = LabelledDataset(
            ml_action.cm.get_test_annotations_path(),
            get_resnet_test_transforms(), return_paths=True)
        testlaoder = DataLoader(
            test_ds, batch_size=ml_action.cm.get_batch_size(),
            shuffle=True, num_workers=0)

        LOG.info('Start testing model')
        acc, loss, predictions, paths = model.test(testlaoder)

        append_to_json_file(
            ml_action.cm.get_test_results_file(),
            {'acc': acc,
             'loss': loss,
             'predictions': predictions.tolist(),
             'paths': paths.tolist()
             })
        with test_image_counter.get_lock():
            test_image_counter.value = 0
        torch.cuda.empty_cache()
        return ActionStatus.SUCCESS, 'Test completed'
    except Exception as e:
        torch.cuda.empty_cache()
        return ActionStatus.FAILED, \
            f'Unknwon error occured durning training ({str(e)})'


def predict(**kwargs):
    try:
        n_predictions = kwargs.get('n_predictions')
        random = kwargs.get('random')
        balance = kwargs.get('balance')
        ml_action = MLAction()
        model = ml_action.load_training_model()
        unl_loader = ml_action.get_unl_loader()

        LOG.info('Start prediction with parameters: '
                 f'random={random}, balance={balance}')
        pm = PredictionsManager(ml_action.cm.get_n_labels(),
                                ml_action.cm.get_predictions_file())

        pm.get_new_predictions(
            n_predictions, model=model,
            dataloader=unl_loader, random=random, balance=balance)
        torch.cuda.empty_cache()
        return ActionStatus.SUCCESS, 'Prediction completed'
    except Exception as e:
        torch.cuda.empty_cache()
        return ActionStatus.FAILED, \
            f'Unknwon error occured durning prediction: ({str(e)})'
