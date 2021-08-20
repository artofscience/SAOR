from .interior_point import InteriorPoint, State
from scipy.sparse import diags
import numpy as np
from copy import deepcopy

"""
This is a primal-dual interior-point approach (without artificial variables)
to solve a convex and separable nonlinear optimization problem.
The problem is solved by applying Newton's method to sequence of relaxed KKT conditions.

Problem P:

min     g0[x]                       (objective)
s.t.    gi[x] <= ri,    i = 1...m   (constraints)
        aj <= xj <= bj  i = 1...n   (bound constraints)

Lagrangian L:

L := g0[x] + sum(lami * (gi[x] - ri)) + sum(alphaj * (aj - xj) + betaj * (xj - bj))
lami    >= 0    := Lagrange multipliers wrt     gi[x] <= ri
alphaj  >= 0    := Lagrange multipliers wrt        aj <= xj
betaj   >= 0    := Lagrange multipliers wrt        xj <= bj

L           = psi[x,lam] - sum(lami * ri) + sum(alphaj * (aj - xj) + betaj * (xj - bj))
psi[x,lam]  = g0[x] + sum(lami * gi[x])

KKT conditions:

psi/dxj - xsij + etaj   =   0       (dL/dxj = 0)

gi[x] - ri              <=  0       (primal feasibility)
aj - xj                 <=  0       (primal feasibility)
xj - bj                 <=  0       (primal feasibility)

etaj                    >=  0       (dual feasibility)
xsij                    >=  0       (dual feasibility)
lami                    >=  0       (dual feasibility)

lami * (gi[x] - ri)     =   0       (complementary slackness)
xsij * (aj - xj)        =   0       (complementary slackness)
etaj * (xj - bj)        =   0       (complementary slackness)

The zeros in the right hand sides of the complementary slackness conditions are replaced by
a "small" negative parameter epsi > 0.

Slack variables yi are introduced for the constraints.

RELAXED KKT conditions:

psi/dxj - xsij + etaj   =   0       (dL/dxj = 0)

gi[x] - ri + si         =   0       (primal feasibility)
xj - aj                 >   0       (primal feasibility)
bj - xj                 >   0       (primal feasibility)

etaj                    >   0       (dual feasibility)
xsij                    >   0       (dual feasibility)
lami                    >   0       (dual feasibility)
si                      >   0

lami * si - epsi        =   0       (complementary slackness)
xsij * (aj - xj) - epsi =   0       (complementary slackness)
etaj * (xj - bj) - epsi =   0       (complementary slackness)

Given a point w = (x,lam,xsi,eta,s) which satisfies feasibility
one can apply Newton's method to obtain dw = (dx,dlam,dxsi,deta,ds).
Here dxsi, deta and ds can be eliminated without severe computational effort.

Subsequently we are left with a reduced system in terms of dx and dlam
"""


class InteriorPointX(InteriorPoint):
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

        """
        Svanberg's implementation uses w.x = (a + b)/2.
        I found w.x = x (that is use the old variable field as initial guess)
        to converge much faster.
        Note however that a < x < b must hold. For variables where this does not hold one should use
        w.x = (a + b)/2
        """

        self.w = State(
            # self.x0,
            (self.beta + self.alpha) / 2,
            np.maximum(1 / (self.x0 - self.alpha), 1),
            np.maximum(1 / (self.beta - self.x0), 1),
            np.ones(self.m),
            np.ones(self.m)
        )

        self.r = deepcopy(self.w)
        self.dw = deepcopy(self.w)
        self.wold = deepcopy(self.w)

    def residual(self):
        """Updates the residual and return its norm and maximum.

        r(x)        = psi / dx - xsi + eta = dg/dx[x] obj + lam' * dg/dx[x] constraints - xsi + eta
        r(xsi)      = xsi * (x - a) - e
        r(eta)      = eta * (b - x) - e
        r(lam)      = gi[x] - ri + si
        r(s)        = lam * si - e
        """
        dg = self.dg(self.w.x)
        self.r.x = dg[0] + self.w.lam.dot(dg[1:]) - self.w.xsi + self.w.eta
        self.r.xsi = self.w.xsi * (self.w.x - self.alpha) - self.epsi
        self.r.eta = self.w.eta * (self.beta - self.w.x) - self.epsi
        self.r.lam = self.g(self.w.x)[1:] + self.w.s
        self.r.s = self.w.lam * self.w.s - self.epsi
        return self.r.norm(), self.r.max()

    def get_newton_direction(self):
        # Some calculations to omit repetitive calculations later on
        a = self.w.x - self.alpha
        b = self.beta - self.w.x
        g = self.g(self.w.x)
        dg = self.dg(self.w.x)
        ddg = self.ddg(self.w.x)

        # delta_lambda
        delta_lambda = g[1:] + self.epsi / self.w.lam
        delta_x = dg[0] + self.w.lam.dot(dg[1:]) - self.epsi / a + self.epsi / b

        diag_lambda = self.w.s / self.w.lam  # s./lam
        diag_x = ddg[0] + self.w.lam.dot(ddg[1:]) + self.w.xsi / a + self.w.eta / b

        if self.m > self.n:
            dldl = delta_lambda / diag_lambda
            B = -delta_x - dldl.dot(dg[1:])
            A = diags(diag_x) + dg[1:].transpose().dot(diags(1 / diag_lambda) * dg[1:])

            # solve for dx
            self.dw.x[:] = np.linalg.solve(A, B)  # n x n
            self.dw.lam[:] = dg[1:].dot(self.dw.x) / diag_lambda + dldl  # calculate dlam[dx]

        else:
            dxdx = delta_x / diag_x
            B = delta_lambda - dxdx.dot(dg[1:].transpose())
            A = diags(diag_lambda) + np.einsum("ki,i,ji->kj", dg[1:], 1 / diag_x, dg[1:])

            # solve for dlam
            self.dw.lam = np.linalg.solve(A, B)  # m x m
            self.dw.x = -dxdx - (self.dw.lam.dot(dg[1:])) / diag_x

        # get dxsi[dx], deta[dx] and ds[dlam]
        self.dw.xsi = -self.w.xsi + self.epsi / a - (self.w.xsi * self.dw.x) / a
        self.dw.eta = -self.w.eta + self.epsi / b + (self.w.eta * self.dw.x) / b
        self.dw.s = -self.w.s + self.epsi / self.w.lam - (self.w.s * self.dw.lam) / self.w.lam
