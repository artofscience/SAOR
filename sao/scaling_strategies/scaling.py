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
        return NotImplementedError

    @abstractmethod
    def update_factor(self, f, df, **kwargs):
        return NotImplementedError

    def scale(self, f=None, df=None, **kwargs):
        if self.update_condition(f, df):
            self.update_factor(f, df)
        return self.factor * f, (df.T * self.factor).T


class InitialObjectiveScaling(Scaling):
    """
    This is an example of an implemented scaling_strategies class.
    Here, the objective is scaled wrt its value at the 0th iteration.
    """

    def __init__(self, nresp, **kwargs):
        """
        Initializes a scaling class that scales the objective wrt to its value at the 0th iteration.

        :param nresp: Number of responses
        :param kwargs: `scale_to` is a vector that holds a desired (scaled) value for each response
        """
        super().__init__(nresp, **kwargs)
        self.scale_to = kwargs.get('scale_to', np.ones_like(self.factor))

    def update_factor(self, f, df, **kwargs):
        if f[0] == 0:
            raise ZeroDivisionError(f'Cannot use {self.__class__.__name__} class with objective value 0')
        self.factor[0] = self.scale_to[0] / f[0]
        return self

    def update_condition(self, f, df, **kwargs):
        if np.all(self.factor == 1):
            return True
        else:
            return False


class InitialResponseScaling(Scaling):
    """
    This class scales the objective and the constraints with respect to their value at the 0th iteration.
    This means that for the vector of responses `f` with f.shape = (m+1,) we have a respective vector `factor`, where:
        factor.shape = (m+1,)
        factor[0] = scale_to / f[0]
        f_scaled[j] = f[j] / norm(df[j, :]) , for j = 1,...,m
    """

    def __init__(self, nresp, **kwargs):
        super().__init__(nresp, **kwargs)
        self.scale_to = kwargs.get('scale_to', np.ones_like(self.factor))

    def update_factor(self, f, df, **kwargs):
        """
        This method updates the scaling factors for all responses,
        i.e. the vector `self.factor` of size (m+1,) is updated.

        :param f: Response vector
        :param df: Sensitivity matrix
        :param kwargs:
        :return: self
        """
        # Scale objective wrt to its value at the 0th iteration
        if f[0] == 0:
            raise ZeroDivisionError(f'Cannot use {self.__class__.__name__} class with objective value 0')
        self.factor[0] = self.scale_to[0] / f[0]

        # Scale constraints wrt the norm of their sensitivities
        for j in range(1, len(f)):
            constr_norm = np.linalg.norm(df[j, :])
            if constr_norm == 0:
                raise ZeroDivisionError(f'Cannot use {self.__class__.__name__} class when ||dg_j/dx|| = 0')
            self.factor[j] = self.scale_to[j] / constr_norm
        return self

    def update_condition(self, f, df, **kwargs):
        if np.all(self.factor == 1):
            return True
        else:
            return False
