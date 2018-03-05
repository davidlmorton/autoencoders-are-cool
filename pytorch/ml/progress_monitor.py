from abc import ABC, abstractmethod
from ml.utils import Progbar
from torch.autograd import Variable


class ProgressMonitor(ABC):
    @abstractmethod
    def step(self, data):
        pass

    @abstractmethod
    def start_session(self, num_steps, metric_names, **kwargs):
        pass


class StdoutProgressMonitor(ProgressMonitor):
    def __init__(self):
        self._progress_bar = None

    def start_session(self, num_steps, metric_names, **kwargs):
        self._metric_names = metric_names
        self._progress_bar = Progbar(num_steps, **kwargs)

    def step(self, data):
        if self._progress_bar is None:
            raise RuntimeError("You must call 'start_session' before "
                    "calling 'step'")

        values = []
        for key, name in self._metric_names.items():
            value = data[key]
            if isinstance(value, Variable):
                value = value.data[0]
            values.append((name, value))
        self._progress_bar.add(1, values)
