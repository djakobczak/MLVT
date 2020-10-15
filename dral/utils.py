from torchvision import transforms


def get_resnet18_default_transforms():
    return transforms.Compose([
        # transforms.Resize(256),
        # transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def get_relative_paths(paths, relative_to):  # can throw an exception
    rpaths = []
    for path in paths:
        idx = path.index(relative_to)
        rpaths.append(path[idx:])
    return rpaths
