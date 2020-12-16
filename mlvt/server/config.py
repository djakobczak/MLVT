import os

# Default config
# CONFIG_NAME = 'testset'
CONFIG_NAME = 'cars_aircrafts'
# CONFIG_NAME = 'sushi_sandwiches'
# CONFIG_NAME = 'audi_vs_bmw'
# CONFIG_NAME = 'male_vs_famale'

DEFAULT_CONFIG = 'cars_aircrafts'

USER_IMAGE_DIR = os.path.join('.', 'mlvt', 'server', 'static', 'user_images')
RELATIVE_USER_IMAGE_DIR = os.path.join('static',  'user_images')
CUT_STATIC_IDX = USER_IMAGE_DIR.index('static')

CONFIG_PATH = os.path.join('.', 'mlvt', 'config', 'config.yml')
CURRENT_CONFIG_FILE = os.path.join('.', 'mlvt', 'config', 'current_config.txt')

PORT = 5000
