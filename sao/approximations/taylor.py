from .approximation import Approximation
import numpy as np


class Taylor1(Approximation):
    def g(self, y=None, dy=None, ddy=None):
        # return self.f + self.P.dot(y-self.y)
        return self.f + (self.P.dot(y-self.y) if len(y.shape) == 1 else np.diag(self.P.dot(y-self.y)))

    def dg(self, y=None, dy=None, ddy=None):
        return self.P * dy

    def ddg(self, y=None, dy=None, ddy=None):
        return self.P * ddy


class Taylor2(Taylor1):
    def g(self, y=None, dy=None, ddy=None):
        return super().g(y) + 0.5 * (self.Q.dot((y-self.y)**2) if len(y.shape) == 1 else np.diag(self.Q.dot(y-self.y)))

    def dg(self, y=None, dy=None, ddy=None):
        return super().dg(y, dy) + (self.Q * (y-self.y).T) * dy

    def ddg(self, y=None, dy=None, ddy=None):
        return super().ddg(y, dy, ddy) + (self.Q*(y-self.y).T)*ddy + self.Q*dy**2
