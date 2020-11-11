import os


# CONFIG_NAME = 'testset'
CONFIG_NAME = 'cars_aircrafts'
# CONFIG_NAME = 'sushi_sandwiches'

USER_IMAGE_DIR = os.path.join('.', 'server', 'static', 'user_images')
RELATIVE_USER_IMAGE_DIR = os.path.join('static',  'user_images')
CUT_STATIC_IDX = USER_IMAGE_DIR.index('static')
