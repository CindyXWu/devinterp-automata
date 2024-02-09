from torch.utils.data import IterableDataset
from typing import Iterable, TypeVar, Optional, Iterator
T = TypeVar("T")

from di_automata.tasks.automata import AutomatonDataset
from di_automata.config_setup import MainConfig, config_class_map


class TorchDatasetFromIterable(IterableDataset):
    def __init__(self, config: MainConfig, deterministic: bool):
        super(TorchDatasetFromIterable, self).__init__()
        self.config, task_config = config, config.task_config
        task_config["seed"] = 42 if deterministic else None
        self.automaton_dataset = AutomatonDataset(task_config)
        config_class = config_class_map[self.config.dataset_type]
        self.task_config_instance = config_class(**task_config)
        
    def __iter__(self):
        while True:
            x, y = self.automaton_dataset.automaton.sample()
            yield {
                "input_ids": x,
                "label_ids": y
            }
    
    def __len__(self):
        """Define steps per epoch to prevent infinite training."""
        return self.task_config_instance.vocab_size ** self.task_config_instance.length
    

def take_n(i: Iterable[T],  n: Optional[int]) -> Iterator[T]:
    """
    Take the first `n` elements from the iterable `i`. If n is None, iterate over all elements.

    The resulting iterator is always finite.
    
    Used to prevent IterableDataset from training forever within an epoch.
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
