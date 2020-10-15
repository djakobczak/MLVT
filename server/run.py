from server import app
from dral.models import init_and_save


if __name__ == '__main__':
    init_and_save(app.app.config['MODEL_PATH'])
    app.run(debug=True)
