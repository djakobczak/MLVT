from flask import Blueprint, render_template


class AnnotationException(Exception):
    pass


class ActionLockedException(Exception):
    pass


class FileException(Exception):
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
def handle_currupted_fille(error):
    return render_template(
        "error.html.j2", code=400, msg=str(error)), 400
