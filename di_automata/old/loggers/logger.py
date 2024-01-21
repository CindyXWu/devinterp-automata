import abc
from typing import Optional, Sequence, Dict


class LoggerBase(abc.ABC):
    """
    A general class for logging training artifacts like metrics and images.

    Should be general enough to be subclassed for different logging backends like Tensorboard, Comet, WandB etc.

    The only required method is log_scalar. The other logging methods, if not implemented by the subclass, will just
    do nothing.
    """
    def __init__(self) -> None:
        super().__init__()
        self._step = 0

    def increment_step(self):
        """
        Increments the internal step counter by 1.
        """
        self._step += 1

    @abc.abstractmethod
    def log_scalar(self, name: str, value: float, step: Optional[int] = None):
        pass

    @abc.abstractmethod
    def log_scalars(self, name: str, value: Sequence[float], step: Optional[int] = None):
        pass
    
    @abc.abstractmethod
    def log_dictionary(self, dictionary: Dict[str, float], step: Optional[int] = None):
        pass


class NullLogger(LoggerBase):
    """
    A logger that does nothing. Useful for default arguments in functions.
    """
    def log_scalar(self, name: str, value: float, step: Optional[int] = None):
        pass

    def log_scalars(self, name: str, value: Sequence[float], step: Optional[int] = None):
        pass
    
    def log_dictionary(self, dictionary: Dict[str, float], step: Optional[int] = None):
        pass