from typing import Any, Dict, Iterable, List, Optional, Protocol, Set, Sequence
import numpy.typing as npt
import warnings
import dataclasses
import logging
import csv
import numpy as np
import pandas as pd

from di_automata.loggers.logger import LoggerBase


class CSVLogger(LoggerBase):
    """Logs data to a CSV file.
    
    As standard for this setup, loggers do not self-increment.
    """

    def __init__(self, out_file: str):
        self.out_file = out_file
        self.metrics = []
        self.buffer = {}

    def add_metrics(self, new_metrics: Iterable[str]):
        """Update the list of metrics to be logged."""
        self.metrics += list(new_metrics)

        try:
            logs = pd.read_csv(self.out_file)
        except FileNotFoundError:
            logs = pd.DataFrame(columns=["step"])

        logs = logs.reindex(columns=list(logs.columns) + list(new_metrics))
        logs.to_csv(self.out_file, index=False)

    def log_scalar(self, name: str, value: float, step: Optional[int] = None):
        """Log a single scalar value."""
        self._log({name: value}, step)
    
    def log_scalars(self, name: str, values: Sequence[float], step: Optional[int] = None):
        """Log multiple scalar values."""
        data = {f"{name}_{i}": value for i, value in enumerate(values)}
        self._log(data, step)
        
    def _log(self, data, step=None, commit=None, **_):
        """Log the data at a specific step.
        
        This function does not increment the step. If the specified step count is greater than the logger internal step count (step is explicitly logged in e.g. multiples of some value > 1), accept. Otherwise, reject the log.
        """
        new_metrics = set(data.keys()) - set(self.metrics)

        if new_metrics:
            self.add_metrics(new_metrics)

        if step is not None:
            if step < self._step:
                warning = f"Step must be greater than {self._step}, got {step}. Ignoring."
                warnings.warn(warning)
                return
            elif step > self._step:
                if self._step >= 0:
                    self._commit()

                self._step = step

        self.buffer.update(data)

        if commit:
            self._commit()

    def _commit(self):
        """Write the buffered data to the CSV file."""
        if self._step < 0:
            return

        with open(self.out_file, "a", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(
                [self._step]
                + [self.buffer.get(metric, None) for metric in self.metrics]
            )
        self.buffer.clear()