from flask import Blueprint, render_template


class AnnotationException(Exception):
    pass


class ActionLockedException(Exception):
    pass


class FileException(Exception):
    pass


class ModelException(Exception):
    pass


class PathException(Exception):
    pass


class EmptyFileException(Exception):
    pass


errors = Blueprint('errors', __name__)

PLAIN = 'ContentType: text/plain'


@errors.app_errorhandler(AnnotationException)
def handle_annotations(error):
    return render_template(
        "error.html.j2", code=400, msg=str(error)), 400


@errors.app_errorhandler(ActionLockedException)
def handle_lock(error):
    return render_template(
        "error.html.j2", code=409, msg=str(error)), 409


@errors.app_errorhandler(FileException)
def handle_currupted_file(error):
    return render_template(
        "error.html.j2", code=400, msg=str(error)), 400


@errors.app_errorhandler(ModelException)
def handle_model(error):
    return render_template(
        "error.html.j2", code=400, msg=str(error)), 400


@errors.app_errorhandler(PathException)
def handle_path(error):
    return render_template(
        "error.html.j2", code=400, msg=str(error)), 400


@errors.app_errorhandler(EmptyFileException)
def handle_empty_file(error):
    return str(error), 400
