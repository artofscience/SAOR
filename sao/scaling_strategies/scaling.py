from abc import ABC, abstractmethod
import numpy as np


class Scaling(ABC):
    """
    This is an abstract implementation of a Scaling class.
    """

    def __init__(self, nresp, **kwargs):
        self.factor = np.ones(nresp, dtype=float)

    @abstractmethod
    def update_condition(self, f, df, **kwargs):
        """This method implements a condition that decides whether re-scaling should be applied."""
        return NotImplementedError

    @abstractmethod
    def update_factor(self, f, df, **kwargs):
        """This method updates the scaling factor, i.e. ``f_scaled = factor * f``."""
        return NotImplementedError

    def scale(self, f=None, df=None, **kwargs):
        """ This method scales the response vector ``f`` and the sensitivity matrix ``df``
        according to the `update_factor` method when ``update_condition`` is met.
        """
        if self.update_condition(f, df):
            self.update_factor(f, df)
        return self.factor * f, (df.T * self.factor).T


