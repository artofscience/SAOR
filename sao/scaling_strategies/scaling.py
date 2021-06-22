from abc import ABC, abstractmethod
import numpy as np


class Scaling(ABC):
    """
    This is the abstract implementation of the scaling_strategies class.
    """
    def __init__(self, **kwargs):
        self.factor = None

    @abstractmethod
    def update_factor(self, **kwargs):
        ...


class ObjectiveScaling(Scaling):
    """
    This is a simple class to scale the objective with respect to its value at the 1st iteration.
    This means that for the vector of responses `f` with f.shape = (m+1,) we have: f_scaled[0] = f[0] / f^(0)[0].
    """
    def __init__(self, threshold=100, **kwargs):
        super().__init__(**kwargs)
        self.threshold = threshold

    def update_factor(self, f=None, **kwargs):
        # Scale objective function at the 0th iteration, then pass
        if self.factor is None:
            self.factor = np.ones_like(f)
            try:
                self.factor[0] = self.threshold / f[0]
            except ZeroDivisionError:
                print('Cannot use `UniformScaling` class with objective value 0')


class ResponseScaling(ObjectiveScaling):
    """
    This is a class to scale the objective with respect to its value at the 1st iteration, i.e.:
        f_scaled[0] = f[0] / f^(0)[0]
    and the constraints so that their Euclidean norm is always equal to 1, i.e.:
        f_scaled[j] = f[j] / norm(df[j, :])  , for all j
    """
    def update_factor(self, f=None, df=None, **kwargs):
        # Scale objective
        super().update_factor(f)

        # Scale constraints at the 0th iteration
        if np.all(self.factor[1:] == 1):
            for j in range(1, len(f)):
                self.factor[j] = 1 / np.linalg.norm(df[j, :])
