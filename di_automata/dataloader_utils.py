"""Combined dataset.util and data_utils from Bruno's original code."""
from typing import Union, Iterable, Iterator, Optional, TypeVar, Generator
import torch
from torch.utils.data import Dataset
from torchvision.datasets import CIFAR10, MNIST
from torch.utils.data import DataLoader

from numpy.random import SeedSequence, Generator, default_rng


def get_input_shape(dataset: Dataset) -> tuple[int, ...]:
    """
    Get the input shape of a dataset. This is useful for constructing the model.

    Assumes a supervised dataset (x, y)
    """
    return dataset[0][0].shape


def get_output_size(dataset: Union[CIFAR10, MNIST]) -> int:
    """
    Get the output size of a dataset. This is useful for constructing the model.

    Assumes a supervised dataset (x, y)
    """
    # TODO: Generalise for text data.
    return len(dataset.classes)


def get_generator_for_worker(seed_seq: SeedSequence) -> Generator:
    """Get a generator for a worker. Useful in multi-process data loading. Will NOT
    give the same rng sequence for different number of workers, but will give the same
    rng sequence if the same number of workers is used."""
    worker_info = torch.utils.data.get_worker_info()
    if worker_info is None:  # single-process data loading
        rng = default_rng(seed_seq)
    else:  # multi-process data loading
        # Generate as many states from the seed sequence as there are workers, and pick
        # the one corresponding to the current worker.
        worker_seed = seed_seq.generate_state(worker_info.num_workers)[worker_info.id]
        rng = default_rng(worker_seed)
    return rng


def get_data_loaders(
    train_dataset: Dataset,
    eval_datasets: dict[str, Dataset],
    train_batch_size: int = 128,
    eval_batch_size: int = 512,
    num_workers: int = 4,
    pin_memory: bool = False,
) -> tuple[DataLoader, dict[str, DataLoader]]:
    """Generic data loader function (don't use for anything beyond cursory CIFAR10 loading)."""
    train_loader = DataLoader(
        train_dataset,
        batch_size=train_batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    eval_loaders = dict()
    for eval_dataset_name, eval_dataset in eval_datasets.items():
        eval_loaders[eval_dataset_name] = DataLoader(
            eval_dataset,
            batch_size=eval_batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
    return train_loader, eval_loaders


T = TypeVar("T")


def ad_infinitum(i: Iterable[T]) -> Iterator[T]:
    while True:
        yield from i


def take_n(i: Iterable[T],  n: Optional[int]) -> Iterator[T]:
    """
    Take the first `n` elements from the iterable `i`. If n is None, iterate over all elements.

    The resulting iterator is always finite.
    """
    assert n is None or n > 0
    try:
        if n is None:
            # If n is None, then we want to iterate over the entire iterable.
            n = len(i)
    except TypeError as e:
        # If the iterable does not have a length, then `n` must be specified (not None)
        raise ValueError(
            "If the iterable does not have a length, then `n` must be specified (not None), otherwise "
            "resulting iterator might be infinite"
        ) from e

    yield from (x for _, x in zip(range(n), i))