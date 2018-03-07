from abc import ABC, abstractmethod

import math
import numpy as np


class RateController(ABC):
    @abstractmethod
    def new_learning_rate(self, step, data):
        pass

    @abstractmethod
    def start_session(self, num_steps):
        pass


class CosineRateController(RateController):
    def start_session(self, num_steps, min_learning_rate=None,
            max_learning_rate=1e-6):
        if min_learning_rate is None:
            min_learning_rate = max_learning_rate / 50.0

        self.min_learning_rate = min_learning_rate
        self.max_learning_rate = max_learning_rate
        self.num_steps = num_steps

    def new_learning_rate(self, step, data):
        if self.num_steps is None:
            raise RuntimeError("You must call 'start_session' before calling "
                    "'new_learning_rate'")
        if step > self.num_steps:
            raise RuntimeError(f"Argument step={step}, should not exceed "
                    f"num_steps={self.num_steps}")

        progression = step / self.num_steps
        new_learning_rate = (self.min_learning_rate +
                (self.max_learning_rate - self.min_learning_rate) *
                (1 + math.cos(math.pi * progression)) / 2)
        return new_learning_rate

class ExponentialRateController(RateController):
    def start_session(self, num_steps, min_learning_rate,
            max_learning_rate):
        self.min_learning_rate = min_learning_rate
        self.max_learning_rate = max_learning_rate
        self.num_steps = num_steps

        log_rates = np.linspace(np.log(min_learning_rate),
                                np.log(max_learning_rate),
                                num_steps)
        self.learning_rates = np.exp(log_rates)

    def new_learning_rate(self, step, data):
        if self.num_steps is None:
            raise RuntimeError("You must call 'start_session' before calling "
                    "'new_learning_rate'")
        if step > self.num_steps:
            raise RuntimeError(f"Argument step={step}, should not exceed "
                    f"num_steps={self.num_steps}")

        new_learning_rate = self.learning_rates[step]
        return new_learning_rate
