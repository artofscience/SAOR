import numpy as np
from .ConvCriterion import ConvergenceCriterion


class KKT(ConvergenceCriterion):
    """
    The KKT conditions as a convergence criterion.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.xmin = kwargs.get('xmin', None)
        self.xmax = kwargs.get('xmax', None)

    def get_response(self, **kwargs):
        """
        Function to calculate |KKT|: Filter out bound constraints so that KKT goes to 0 when convergence is achieved.

        :param kwargs:
        :return:
        """

        x_k = kwargs.get('x_k', None)
        df = kwargs.get('df', None)
        lam = kwargs.get('lam', None)
        KKT_red_idx = [index for index, value in enumerate(x_k)
                       if 1.001 * self.xmin[index] < value < 0.99 * self.xmax[index]]
        KKT_reduced = df[0, KKT_red_idx] + np.dot(lam.T, df[1:, KKT_red_idx])
        KKT_res = np.linalg.norm(KKT_reduced)
        return KKT_res