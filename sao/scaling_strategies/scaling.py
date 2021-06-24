from abc import ABC, abstractmethod
import numpy as np


class Scaling(ABC):
    """
    This is the abstract implementation of the Scaling class.
    """
    def __init__(self, **kwargs):
        self.factor = None

    @abstractmethod
    def update(self, **kwargs):
        ...


# TODO: This can be improved, since f[0] = 100 always as it is rescaled every iteration (probably redundant).
#  Maybe we can rescale only when if f[0] < 1 or something similar.
class ObjectiveScaling(Scaling):
    """
    This is a class that scales the objective with respect to its value at the current iteration.
    This means that for the vector of responses `f` with f.shape = (m+1,) we have a respective vector `factor`, where:
        factor.shape = (m+1,)
        factor[0] = scale_to / f[0]
        factor[1:] = 1
    """

    def __init__(self, scale_obj_to=100, **kwargs):
        super().__init__(**kwargs)
        self.scale_obj_to = scale_obj_to


    def update(self, f=None, df=None, **kwargs):
        """
        This method updates the scaling factor for the objective.

        :param f: Response vector
        :param df: Sensitivity matrix
        :param kwargs:
        :return: self
        """

        if self.factor is None:
            self.factor = np.ones_like(f)
        if f[0] == 0:
            raise ZeroDivisionError(f'Cannot use {self.__class__.__name__} class with objective value 0')
        self.factor[0] = self.scale_obj_to / f[0]

        return self


# TODO: This can be improved, as f[0] = 100 always and f[1:] are rescaled every iteration (probably redundant).
class ResponseScaling(ObjectiveScaling):
    """
    This is a simple Scaling class that scales the objective and the constraints with respect to their value
    at the current iteration. This means that for the vector of responses `f` with f.shape = (m+1,) we have a
    respective vector `factor`, where:
        factor.shape = (m+1,)
        factor[0] = scale_to / f[0]
        f_scaled[j] = f[j] / norm(df[j, :]) , for j = 1,...,m
    """

    def __init__(self, scale_obj_to=100, scale_constr_to=1, **kwargs):
        super().__init__(scale_obj_to, **kwargs)
        self.scale_constr_to = scale_constr_to

    def update(self, f=None, df=None, **kwargs):
        """
        This method updates the scaling factor for each response (only at the 0th iteration).

        :param f: Response vector
        :param df: Sensitivity matrix
        :param kwargs:
        :return: self
        """

        # Scale objective
        super().update(f)

        # Scale constraints
        for j in range(1, len(f)):
            constr_norm = np.linalg.norm(df[j, :])
            if constr_norm == 0:
                raise ZeroDivisionError(f'Cannot use {self.__class__.__name__} class when ||dg_j/dx|| = 0')
            self.factor[j] = self.scale_constr_to / constr_norm

        return self


class InitialObjectiveScaling(ObjectiveScaling):
    """
    This is a simple Scaling class that scales the objective with respect to its value at the 0th iteration.
    This means that for the vector of responses `f` with f.shape = (m+1,) we have a respective vector `factor`, where:
        factor.shape = (m+1,)
        factor[0] = scale_to / f[0]
        factor[1:] = 1
    """

    def update(self, f=None, df=None, **kwargs):
        """
        This method updates the scaling factor for the objective (only at the 0th iteration).

        :param f: Response vector
        :param df: Sensitivity matrix
        :param kwargs:
        :return: self
        """

        if self.factor is None:
            super().update(f)

        return self


class InitialResponseScaling(ResponseScaling):
    """
    This class scales the objective and the constraints with respect to their value at the 0th iteration.
    This means that for the vector of responses `f` with f.shape = (m+1,) we have a respective vector `factor`, where:
        factor.shape = (m+1,)
        factor[0] = scale_to / f[0]
        f_scaled[j] = f[j] / norm(df[j, :]) , for j = 1,...,m
    """

    def update(self, f=None, df=None, **kwargs):
        """
        This method updates the scaling factor for each response (only at the 0th iteration).

        :param f: Response vector
        :param df: Sensitivity matrix
        :param kwargs:
        :return: self
        """

        # Scale responses at 0th iteration
        if self.factor is None:
            super().update(f, df)

        return self
