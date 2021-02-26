from .primal_dual import PrimalDual
import numpy as np
from abc import ABC, abstractmethod


class InteriorPoint(PrimalDual, ABC):
    """
    Primal-dual interior point method.
    Construction provides the problem object which (at least) contains:
    g[x] (m+1 x 1)          Responses
    dg[x] (n x m+1)         Sensitivities
    ddg[x] (n x m+1)        (optionally 2nd order diagonal sensitivities)
    r (m+1 x 1)             zero order terms

    In addition it provides the current design point (x) and bounds (a and b)
    """

    def __init__(self, problem, **kwargs):
        super().__init__(problem)

        self.epsimin = kwargs.get('epsimin', 1e-6)
        self.iteramax = kwargs.get('iteramax', 50)
        self.iterinmax = kwargs.get('iterinmax', 100)
        self.alphab = kwargs.get('alphab', -1.01)
        self.epsifac = kwargs.get('epsifac', 0.9)
        self.epsired = kwargs.get('epsired', 0.5)  # 0.0 < (float) epsired < 1.0

        """
        Initialization of variables, old variables, variable step and residual vectors
        w = [x, lambda, xsi, eta, s]
        r = [rx, rlambda, rxsi, reta, rs]
        dw = [dx, dlambda, dxsi, deta, ds]
        dwold = [dxold, dlambdaold, dxsiold, detaold, dsold]
        """

        self.iterout = 0
        self.iterin = 0
        self.itera = 0
        self.step = 0
        self.epsi = 1

    r: list = NotImplemented
    w: list = NotImplemented
    dw: list = NotImplemented
    wold: list = NotImplemented

    @abstractmethod
    def get_residual(self):
        ...

    @abstractmethod
    def get_newton_direction(self):
        ...

    @abstractmethod
    def get_step_size(self):
        ...

    def update(self):

        # iterate until convergence
        while self.epsi > self.epsimin:

            # Calculate the initial residual, its norm and maximum value
            # This indicates how far we are from the global optimum for THIS epsi
            self.get_residual()
            rnorm = np.linalg.norm([np.linalg.norm(i) for i in self.r])
            rmax = np.max([np.max(i) for i in self.r])

            self.iterin = 0
            while rmax > self.epsifac*self.epsi and self.iterin < self.iterinmax:
                self.iterin += 1
                self.iterout += 1


                """
                Get the Newton direction
                This basically builds dw and includes a solve
                """
                self.get_newton_direction()

                # Set w_old = w
                for count, value in enumerate(self.wold):
                    value[:] = self.w[count]

                # Initialize the counter for the line search
                self.itera = 0
                rnew = 2*rnorm

                # Line search in the Newton direction dw
                while rnew > rnorm and self.itera < self.iteramax:
                    self.itera += 1

                    # calculate step size
                    self.get_step_size()

                    # set a step in the Newton direction w^(l+1) = w^(l) + step^(l) * dw
                    for count, value in enumerate(self.w):
                        value[:] = self.wold[count] + self.step * self.dw[count]

                    self.get_residual()
                    rnew = np.linalg.norm([np.linalg.norm(i) for i in self.r])
                    self.step *= 0.5

                rnorm = 1.0 * rnew
                rmax = np.max([np.max(i) for i in self.r])
                self.step *= 2

            self.epsi *= self.epsired


        # end
