LABEL_MAPPING_PETS = {
    0: 'Cat',
    1: 'Dog'
}

LABEL_MAPPING_RPS = {
    0: 'Rock',
    1: 'Paper',
    2: 'Scissors'
}

CONFIG_PETIMAGES64_NORM = {
    'img_size': 64,
    'labels': {
        'data/PetImages/Cat': 0,
        'data/PetImages/Dog': 1
    },
    'n_train': 8000,
    'n_eval': 2000,
    'n_test': 2000,
    'data': {
        'x_path': 'data/x_cats_dogs_64_norm.npy',
        'y_path': 'data/y_cats_dogs_64_norm.npy'
    },
    'max_reward': 5,
    'max_queries': 100,
    'query_punishment': 0.5,
    'left_queries_punishment': 5,
    'reward_treshold': 0.92,
    'reward_multiplier': 4,
}


CONFIG_PETIMAGES64 = {
    'img_size': 64,
    'labels': {
        'data/PetImages/Cat': 0,
        'data/PetImages/Dog': 1
    },
    'n_train': 5000,
    'n_eval': 2000,
    'n_test': 2000,
    'data': {
        'x_path': 'data/x_cats_dogs.npy',
        'y_path': 'data/y_cats_dogs.npy'
    },
    'max_reward': 5,
    'max_queries': 30,
    'query_punishment': 0.5,
    'left_queries_punishment': 5,
    'reward_treshold': 0.92,
    'reward_multiplier': 4,
}

CONFIG_PETIMAGES128 = {
    'img_size': 128,
    'labels': {
        'data/PetImages/Cat': 0,
        'data/PetImages/Dog': 1
    },
    'n_train': 10000,
    'n_eval': 2000,
    'n_test': 2000,
    'data': {
        'x_path': 'data/x_cats_dogs_128.npy',
        'y_path': 'data/y_cats_dogs_128.npy'
    },
    'max_reward': 5,
    'max_queries': 30,
    'query_punishment': 0.5,
    'left_queries_punishment': 5,
    'reward_treshold': 0.92,
    'reward_multiplier': 4,
}

CONFIG_RPS = {
    'img_size': 64,
    'labels': {
        'data/RPS_dataset/rock': 0,
        'data/RPS_dataset/paper': 1,
        'data/RPS_dataset/scissors': 2
    },
    'n_train': 1200,
    'n_eval': 0,
    'n_test': 800,
    'data': {
        'x_path': 'data/x_rps_skimage_range255.npy',
        'y_path': 'data/y_rps_skimage_range255.npy',
    },
    'max_reward': 5,
    'max_queries': 50,
    'query_punishment': 0.5,
    'left_queries_punishment': 5,
    'reward_treshold': 0.7,
    'reward_multiplier': 4,
}

CONFIG_PETIMAGES96 = {
    'img_size': 96,
    'labels': {
        'data/PetImages/Cat': 0,
        'data/PetImages/Dog': 1
    },
    'n_train': 10000,
    'n_eval': 0,
    'n_test': 1000,
    'data': {
        'x_path': 'data/x_cats_dogs_96.npy',
        'y_path': 'data/y_cats_dogs_96.npy'
    },
    'max_reward': 5,
    'max_queries': 30,
    'query_punishment': 0.5,
    'left_queries_punishment': 5,
    'reward_treshold': 0.92,
    'reward_multiplier': 4,
}

CONFIG = CONFIG_PETIMAGES64_NORM
LABEL_MAPPING = LABEL_MAPPING_PETS
