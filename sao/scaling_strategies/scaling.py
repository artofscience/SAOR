from abc import ABC, abstractmethod
import numpy as np


class Scaling(ABC):
    """
    This is the abstract implementation of the scaling_strategies class.
    """
    def __init__(self, *args, **kwargs):
        self.factor = None

    @abstractmethod
    def update_factor(self, *args, **kwargs):
        ...


class ObjectiveScaling(Scaling):
    """
    This is a simple class to scale the objective with respect to its value at the 1st iteration.
    This means that for the vector of responses `f` with f.shape = (m+1,) we have: f_scaled[0] = f[0] / f^(0)[0].
    """
    def __init__(self, threshold=100, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.threshold = threshold

    def update_factor(self, f, *args, **kwargs):
        if self.factor is None:
            self.factor = np.ones_like(f)

            # Scale objective
            try:
                self.factor[0] = self.threshold / f[0]
            except ZeroDivisionError:
                print('Cannot use `UniformScaling` class with objective value 0')
