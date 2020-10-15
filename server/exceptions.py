from flask import Blueprint


class AnnotationException(Exception):
    pass


errors = Blueprint('errors', __name__)

PLAIN = 'ContentType: text/plain'


@errors.app_errorhandler(AnnotationException)
def handle_annotations(error):
    return str(error), 400
