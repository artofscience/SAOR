from .ConvCriterion import ConvergenceCriterion


class ObjectiveChange(ConvergenceCriterion):
    """Function to calculate the normalized change in the objective function values. """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.f, self.fold1 = None, None

    def get_response(self, **kwargs):
        """ A method that calculates the response of the current convergence criterion.

        :param kwargs: Gets current response vector -g- as well as the one of the previous iteration -gold1-
        :return: ObjChange_abs : Normalized change in the objective function values
        """

        self.fold1 = self.f
        self.f = kwargs.get('f', None)

        if self.fold1 is not None:                                        # TODO: Maybe its better with try-except
            if abs(self.fold1[0]) > 1e-5:
                ObjChange_abs = abs((self.f[0] - self.fold1[0]) / self.fold1[0])
            else:
                ObjChange_abs = abs((self.f[0] - self.fold1[0]))
        else:
            ObjChange_abs = 1.0       # large value in order to continue loop
        self.value = ObjChange_abs
        return ObjChange_abs
