import os

import connexion
from connexion.resolver import MethodViewResolver

from server.exceptions import errors


app = connexion.FlaskApp(__name__, port=5000, specification_dir='openapi/')
app.add_api('openapi.yml', strict_validation=True, validate_responses=True,
            resolver=MethodViewResolver('views'))
app.app.config['IMGS_DIR'] = os.path.join('static', 'imgs')
app.app.config['MODEL_PATH'] = os.path.join('data', 'saved_models', 'test_model.pt')
app.app.register_blueprint(errors)
