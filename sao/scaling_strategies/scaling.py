from abc import ABC, abstractmethod
import numpy as np


class Scaling(ABC):
    """
    This is an abstract implementation of a Scaling class.
    """

    def __init__(self, nresp, **kwargs):
        self.factor = np.ones(nresp, dtype=float)

    @abstractmethod
    def update_condition(self, **kwargs):
        return NotImplementedError

    @abstractmethod
    def update_factor(self, f, df, **kwargs):
        return NotImplementedError

    def scale(self, f=None, df=None, **kwargs):
        if self.update_condition():
            self.update_factor(f, df)
        return self.factor * f, (df.T * self.factor).T


class InitialObjectiveScaling(Scaling):
    """
    This is an example of an implemented scaling_strategies class.
    Here, the objective is scaled wrt its value at the 0th iteration.
    """

    def __init__(self, nresp, **kwargs):
        super().__init__(nresp, **kwargs)
        self.scale_to = np.ones_like(self.factor)

    def update_factor(self, f, df, **kwargs):
        if f[0] == 0:
            raise ZeroDivisionError(f'Cannot use {self.__class__.__name__} class with objective value 0')
        self.factor[0] = self.scale_to[0] / f[0]
        return self

    def update_condition(self):
        if np.all(self.factor == 1):
            return True
        else:
            return False
