from torchvision import transforms
import torch


# https://github.com/pratogab/batch-transforms
class ToTensor:
    """Applies the :class:`~torchvision.transforms.ToTensor`
    transform to a batch of images.
    """

    def __init__(self):
        self.max = 255

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor of size (N, C, H, W) to be tensorized.
        Returns:
            Tensor: Tensorized Tensor.
        """
        return tensor.float().div_(self.max)


class Normalize:
    """Applies the :class:`~torchvision.transforms.Normalize` transform to
    a batch of images.
    .. note::
        This transform acts out of place by default, i.e., it does not mutate
        the input tensor.
    Args:
        mean (sequence): Sequence of means for each channel.
        std (sequence): Sequence of standard deviations for each channel.
        inplace(bool,optional): Bool to make this operation in-place.
        dtype (torch.dtype,optional): The data type of tensors to which
        the transform will be applied.
        device (torch.device,optional): The device of tensors to which
        the transform will be applied.
    """

    def __init__(self, mean, std, inplace=False, dtype=torch.float, device='cpu'):
        self.mean = torch.as_tensor(
            mean, dtype=dtype, device=device)[None, :, None, None]
        self.std = torch.as_tensor(
            std, dtype=dtype, device=device)[None, :, None, None]
        self.inplace = inplace

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor of size (N, C, H, W) to be normalized.
        Returns:
            Tensor: Normalized Tensor.
        """
        if not self.inplace:
            tensor = tensor.clone()

        tensor.sub_(self.mean).div_(self.std)
        return tensor


def get_before_tensor_transforms():
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
    ])


def get_resnet_test_transforms():
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]),
    ])


def get_resnet_train_transforms():
    return transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]),
    ])


def get_resnet18_batch_transforms(device):
    return transforms.Compose([
        ToTensor(),
        Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
            device=device),
    ])


def get_relative_paths(paths, relative_to):  # can throw an exception
    rpaths = []
    for path in paths:
        idx = path.index(relative_to)
        rpaths.append(path[idx:])
    return rpaths
