import os

import connexion
from connexion.resolver import MethodViewResolver
from flask import render_template

from mlvt.server.exceptions import errors
from mlvt.server.extensions import executor
from mlvt.server.file_utils import clear_dir
from mlvt.server.config import USER_IMAGE_DIR, PORT
from mlvt.logger import Logger
from connexion.exceptions import ExtraParameterProblem, BadRequestProblem


def handle_404(exception):
    return render_template(
        "error.html.j2", code=404,
        msg="The page you are looking for was not found."), 404


def handle_400(exception):
    return render_template(
        "error.html.j2", code=400,
        msg=f"{exception}"), 400


def bad_request_error_handler(error):
    return render_template(
        "error.html.j2", code=400,
        msg="Bad request"), 400


def register_extensions(app):
    executor.init_app(app)


def register_errors(app):
    app.add_error_handler(404, handle_404)
    app.add_error_handler(400, handle_400)
    app.add_error_handler(ExtraParameterProblem, bad_request_error_handler)
    app.add_error_handler(BadRequestProblem, bad_request_error_handler)


def create_app():
    app = connexion.FlaskApp(__name__, port=PORT, specification_dir='openapi/')
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
    flask_app.config['logger'] = Logger.create_logger('MLVT')
    register_extensions(flask_app)
    clear_dir(USER_IMAGE_DIR)
    register_errors(app)
    return app


if __name__ == "__main__":
    app = create_app()
    app.run(debug=True)
