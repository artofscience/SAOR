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


class ObjectiveScaling(Scaling):
    """
    This is a simple Scaling class that scales the objective with respect to its value at the 0th iteration.
    This means that for the vector of responses `f` with f.shape = (m+1,) we have a respective vector `factor`, where:
        factor.shape = (m+1,)
        factor[0] = threshold / f^(0)[0]
        factor[1:] = 1
    """

    def __init__(self, threshold=100, **kwargs):        # TODO: Don't like much the `threshold` name. Any ideas?
        super().__init__(**kwargs)
        self.threshold = threshold

    def update(self, f=None, df=None, **kwargs):
        """
        This method updates the scaling factor for the objective (only at the 0th iteration).

        :param f: Response vector
        :param df: Sensitivity matrix
        :param kwargs:
        :return:
        """

        if self.factor is None:
            self.factor = np.ones_like(f)
            try:
                self.factor[0] = self.threshold / f[0]
            except ZeroDivisionError:
                print(f'Cannot use {self.__class__.__name__} class with objective value 0')

        return self


class ResponseScaling(ObjectiveScaling):
    """
    This is a simple Scaling class that scales the objective and the constraints with respect to their value
    at the 0th iteration. This means that for the vector of responses `f` with f.shape = (m+1,) we have a
    respective vector `factor`, where:
        factor.shape = (m+1,)
        factor[0] = threshold / f^(0)[0]
        f_scaled[j] = f[j] / norm(df[j, :]) , for j = 1,...,m
    """

    def update(self, f=None, df=None, **kwargs):
        """
        This method updates the scaling factor for each response (only at the 0th iteration).

        :param f: Response vector
        :param df: Sensitivity matrix
        :param kwargs:
        :return:
        """

        # Scale objective
        super().update(f)

        # Scale constraints at the 0th iteration
        if np.all(self.factor[1:] == 1):
            for j in range(1, len(f)):
                self.factor[j] = 1 / np.linalg.norm(df[j, :])

        return self


# TODO: This can be improved, as f[0] = 100 always, as it is rescaled every iteration (probably redundant).
#  Maybe we can rescale only when if f[0] < 1 or something similar.
class IterativeObjectiveScaling(ObjectiveScaling):
    """
    This is a simple Scaling class that scales the objective with respect to its value at the current iteration.
    This means that for the vector of responses `f` with f.shape = (m+1,) we have a respective vector `factor`, where:
        factor.shape = (m+1,)
        factor[0] = threshold / f[0]
        factor[1:] = 1
    """

    def __init__(self, threshold=100, **kwargs):        # TODO: Don't like much the `threshold` name. Any ideas?
        super().__init__(**kwargs)
        self.threshold = threshold


    def update(self, f=None, df=None, **kwargs):
        """
        This method updates the scaling factor for the objective.

        :param f: Response vector
        :param df: Sensitivity matrix
        :param kwargs:
        :return:
        """

        if self.factor is None:
            self.factor = np.ones_like(f)
        try:
            self.factor[0] = self.threshold / f[0]
        except ZeroDivisionError:
            print(f'Cannot use {self.__class__.__name__} class with objective value 0')

        return self


# TODO: This can be improved, as f[0] = 100 always and f[1:] are rescaled every iteration (probably redundant).
class IterativeResponseScaling(IterativeObjectiveScaling):
    """
    This is a simple Scaling class that scales the objective and the constraints with respect to their value
    at the current iteration. This means that for the vector of responses `f` with f.shape = (m+1,) we have a
    respective vector `factor`, where:
        factor.shape = (m+1,)
        factor[0] = threshold / f[0]
        f_scaled[j] = f[j] / norm(df[j, :]) , for j = 1,...,m
    """

    def update(self, f=None, df=None, **kwargs):
        """
        This method updates the scaling factor for each response (only at the 0th iteration).

        :param f: Response vector
        :param df: Sensitivity matrix
        :param kwargs:
        :return:
        """

        # Scale objective
        super().update(f)

        # Scale constraints
        for j in range(1, len(f)):
            self.factor[j] = 1 / np.linalg.norm(df[j, :])

        return self
