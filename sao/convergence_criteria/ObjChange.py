from .ConvCriterion import ConvergenceCriterion


class ObjectiveChange(ConvergenceCriterion):
    """Function to calculate the normalized change in the objective function values. """

    def get_response(self, **kwargs):
        """ A method that calculates the response of the current convergence criterion.

        :param kwargs: Gets current response vector -g- as well as the one of the previous iteration -gold1-
        :return: ObjChange_abs : Normalized change in the objective function values
        """

        g = kwargs.get('g', None)
        gold1 = kwargs.get('gold1', None)
        if gold1 is not None:                                                   # TODO: Maybe its better with try-except
            if abs(gold1[0]) > 1e-5:
                ObjChange_abs = abs((g[0] - gold1[0]) / gold1[0])
            else:
                ObjChange_abs = abs((g[0] - gold1[0]))
        else:
            ObjChange_abs = 1       # large value in order to continue loop
        return ObjChange_abs
