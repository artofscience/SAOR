from sao.solvers.gradient_based_optimizer import GBOpt
from state import State
from abc import ABC, abstractmethod
from copy import deepcopy


class InteriorPoint(GBOpt, ABC):
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
        super().__init__(problem, **kwargs)

        self.epsimin = kwargs.get('epsimin', 1e-6)
        self.max_outer_iter = kwargs.get('outer_iter', 100)
        self.max_lines_iter = kwargs.get('iteramax', 20)
        self.max_inner_iter = kwargs.get('iterinmax', 20)
        self.alphab = kwargs.get('alphab', -1.01)
        self.epsifac = kwargs.get('epsifac', 0.9)
        self.epsired = kwargs.get('epsired', 0.1)  # 0.0 < (float) epsired < 1.0

        """
        Initialization of variables, old variables, variable step and residual vectors
        w = [x, lambda, xsi, eta, s]
        r = [rx, rlambda, rxsi, reta, rs]
        dw = [dx, dlambda, dxsi, deta, ds]
        dwold = [dxold, dlambdaold, dxsiold, detaold, dsold]
        """
        self.iter = 0
        self.iterout = 0
        self.iterin = 0
        self.step = 0
        self.epsi = 1

    r: State = NotImplemented
    w: State = NotImplemented
    dw: State = NotImplemented
    wold: State = NotImplemented

    @abstractmethod
    def residual(self):
        ...

    @abstractmethod
    def get_newton_direction(self):
        ...

    def get_step_size(self):
        step_x = [max(self.alphab * dw/w) for w, dw in zip(tuple(self.w)[1:], tuple(self.dw)[1:])]
        step_alpha = max(self.alphab * self.dw.x / (self.w.x - self.alpha))
        step_beta = max(-self.alphab * self.dw.x / (self.beta - self.w.x))
        return 1.0 / max(1.0, max(step_x), step_alpha, step_beta)

    def update(self):
        self.iter = 0

        while self.iter < self.max_outer_iter and self.epsi > self.epsimin:
            self.iter += 1

            # The initial residual: its norm and maximum. This gives an
            # indicates how far we are from the global optimum for the
            # current ``self.epsi``.
            rnorm, rmax = self.residual()

            inner_iter = 0
            while inner_iter < self.max_inner_iter and rmax > self.epsifac * self.epsi:
                inner_iter = inner_iter + 1

                # Obtain the Newton direction: builds ``dw``
                # Note: this requires solving a system of equations
                self.get_newton_direction()

                self.wold = deepcopy(self.w)  # TODO @max how it this handled?
                # self.wold = self.w # why does this give same result?

                # the setp size can be evaluated once ``dw`` is evaluated
                step = self.get_step_size()

                # line search along the newton direction
                lines_iter = 0
                rnew = 2*rnorm
                while lines_iter < self.max_lines_iter and rnew > rnorm:
                    lines_iter += 1

                    # set a step in the Newton direction
                    # w^(l+1) = w^(l) + step^(l) * dw
                    self.w = self.wold + step * self.dw

                    rnew, rmax = self.residual()
                    step /= 2

                rnorm = rnew
                step *= 2

            self.epsi *= self.epsired

        return self.w.x
