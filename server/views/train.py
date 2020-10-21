from flask import jsonify

from server.views.base import MLView


# !TODO add context manager for load/save flow
class TrainView(MLView):
    def search(self):
        model = self.load_model()
        losses, accs = model.train(
            self.get_train_loader(),
            self.cm.get_epochs())
        print(f'[DEBUG] losses: {losses}, accs: {accs}')
        self.save_model(model.model_conv)
        return jsonify(losses=losses, accuracy=accs)
