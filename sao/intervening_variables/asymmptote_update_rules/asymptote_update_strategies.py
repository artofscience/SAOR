import numpy as np
from abc import ABC, abstractmethod

class AsymptoteUpdateStrategy(ABC):

    @abstractmethod
    def get_asymptotes(self):
        ...

class Svanberg2002(AsymptoteUpdateStrategy):
    def __init__(self, x_min=0.0, x_max=1.0, asyinit=0.5, asyincr=1.2, asydecr=0.7, asybound=10.0, oscillation_tol=1e-10):
        self.xold1, self.xold2 = None, None
        self.x_min, self.x_max = x_min, x_max
        self.x = None

        self.asybound = asybound
        self.asyinit = asyinit
        self.asyincr = asyincr
        self.asydecr = asydecr
        self.oscillation_tol = oscillation_tol

        # MMA parameter initialization
        self.factor = None
        self.dx = self.x_max - self.x_min

        self.min_factor, self.max_factor = 1 / (self.asybound ** 2), self.asybound

    def get_asymptotes(self, x):
        self.xold2, self.xold1, self.x = self.xold1, self.x, x.copy()
        """Increases or decreases the asymptotes interval based on oscillations in the design vector"""
        if self.factor is None:
            self.factor = np.full_like(self.x, self.asyinit)

        if self.xold2 is None:
            # Initial values of asymptotes
            low = x - self.factor * self.dx
            upp = x + self.factor * self.dx
        else:
            # Update scheme for asymptotes
            # depending on if the signs of (x_k-xold) and (xold-xold2) are opposite, indicating an oscillation in xi
            # if the signs are equal the asymptotes are slowing down the convergence and should be relaxed

            # check for oscillations in variables (if > 0: no oscillations, if < 0: oscillations)
            oscillation = ((x - self.xold1) * (self.xold1 - self.xold2)) / self.dx

            # oscillating variables x_i are increase or decrease the factor
            self.factor[oscillation > +self.oscillation_tol] *= self.asyincr
            self.factor[oscillation < -self.oscillation_tol] *= self.asydecr

            # Clip the asymptote factor
            np.clip(self.factor, self.min_factor, self.max_factor)

            # update lower and upper asymptotes
            low = x - self.factor * self.dx
            upp = x + self.factor * self.dx

        return low, upp

