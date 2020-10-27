from dral.logger import LOG
import time

from server.actions.base import MLAction
from server.file_utils import append_to_json_file


def train():
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
        {time.time():
            {'loss': losses,
                'acc': accs}
         })
    return "DONE"
