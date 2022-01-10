import numpy as np

from sao.scaling_strategies import Scaling


class InitialObjectiveScaling(Scaling):
    """
    This is an example of an implemented ``Scaling`` class.
    Here, the objective is scaled wrt its value at the 0th iteration.
    """

    def __init__(self, nresp, **kwargs):
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


class InitialResponseScaling(InitialObjectiveScaling):
    """
    This class scales the objective and the constraints with respect to their value at the 0th iteration.
    This means that for the vector of responses `f` with f.shape = (m+1,) we have a respective vector `factor`, where:
        factor.shape = (m+1,)
        factor[0] = scale_to[0] / f[0]
        factor[j] = scale_to[j] / norm(df[j, :]) , for j = 1,...,m
        f_scaled[j] = factor[j] * f[j]           , for j = 0,...,m
    """

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
