from di_automata.tasks.automata import AutomatonDataset
from torch.utils.data import IterableDataset
from typing import Iterable, TypeVar, Optional, Iterator
T = TypeVar("T")


class TorchDatasetFromIterable(IterableDataset):
    def __init__(self, automaton_dataset: AutomatonDataset):
        super(TorchDatasetFromIterable, self).__init__()
        self.automaton_dataset = automaton_dataset
    
    def __iter__(self):
        return self.automaton_dataset._generate_examples()
    
    def __len__(self):
        """Define steps per epoch to prevent infinite training."""
        return self.automaton_dataset.config.eval_frequency
    

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
