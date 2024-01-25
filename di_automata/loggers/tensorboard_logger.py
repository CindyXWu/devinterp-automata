"""To use with PyTorch, import TensorBoardX."""
from typing import Optional
from di_automata.loggers.logger import LoggerBase
from torch.utils.tensorboard.writer import SummaryWriter


class TensorBoardLogger(LoggerBase):
    def __init__(self, writer: SummaryWriter) -> None:
        super().__init__()
        self._writer = writer
    
    def log_scalar(self, name: str, value: float, step: Optional[int] = None):
        self._writer.add_scalar(name, value, step or self._step)

    def __del__(self):
        self._writer.close()