import os

import connexion
from connexion.resolver import MethodViewResolver

from server.exceptions import errors
from server.extensions import executor
from server.file_utils import clear_dir
from server.config import USER_IMAGE_DIR


def register_extensions(app):
    executor.init_app(app)


def create_app():
    app = connexion.FlaskApp(__name__, port=5000, specification_dir='openapi/')
    app.add_api('openapi.yml', strict_validation=True, validate_responses=True,
                resolver=MethodViewResolver('views'))
    flask_app = app.app
    flask_app.config['IMGS_DIR'] = os.path.join('static', 'imgs')
    flask_app.config['SECRET_KEY'] =  \
        '4b8475762fe5dd0b83f1f5b26588d983fd37e8f8786e399cd37dcb7d2b4b8509'
    flask_app.register_blueprint(errors)
    flask_app.config['EXECUTOR_TYPE'] = 'process'
    flask_app.config['EXECUTOR_MAX_WORKERS'] = 2
    flask_app.config['EXECUTOR_PROPAGATE_EXCEPTIONS'] = True
    register_extensions(flask_app)
    clear_dir(USER_IMAGE_DIR)
    return app


if __name__ == "__main__":
    app = create_app()
    app.run(debug=True)
