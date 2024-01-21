from os import PathLike
from typing import Optional, Sequence
from torch.utils.data import Dataset, Subset
from torchvision.datasets import CIFAR10
from torchvision import transforms


def cifar10_constructor(
    root: PathLike,
    use_data_augmentation: bool = False,
    subset_idxs: Optional[Sequence[int]] = None,
) -> tuple[Dataset, dict[str, Dataset]]:
    base_transforms = [
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]
    test_transforms = transforms.Compose(base_transforms)
    train_transforms = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4),
        ]
        + base_transforms
    ) if use_data_augmentation else test_transforms
    train_set = CIFAR10(root=str(root), train=True, download=True, transform=train_transforms)
    test_set = CIFAR10(root=str(root), train=False, download=True, transform=test_transforms)
    if subset_idxs is not None:
        train_set = Subset(train_set, subset_idxs)
        test_set = Subset(test_set, subset_idxs)
    return train_set, {"test": test_set}