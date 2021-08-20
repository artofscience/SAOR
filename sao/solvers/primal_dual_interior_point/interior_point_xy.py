from .interior_point_x import InteriorPointX
from state import StateY
from scipy.sparse import diags
import numpy as np
from copy import deepcopy

"""
This is an addition to pdip, which extends problem P

Problem P:

min     g0[x]                       (objective)
s.t.    gi[x] <= ri,    i = 1...m   (constraints)
        aj <= xj <= bj  i = 1...n   (bound constraints)

and includes artificial variables y (m x 1) and z (1).


Consider the enriched Part

Problem Part:

min     g0[x]   sum(ci * yi) (objective)
s.t.    gi[x] - yi <= ri,    i = 1...m         (constraints)
        aj <= xj <= bj  i = 1...n                       (bound constraints)
        yi >= 0                                         (bound constraints)

Constant real numbers:
ci >= 0

Further, let di = 1 and ci = "large", so that variables yi become relatively expensive.
Typically, y = 0 in any optimal solution of Part, and the corresponding x is an optimal solution of P as well.

The user should avoid "extremely large" value for ci (e.g. 10^10).
It is good practice to start with low values for ci (e.g. 10^3)
and raise if not all yi* = 0. (star denotes optimal solution)


Lagrangian L:

L := g0[x] + sum(ci*yi) +
    sum(lami * (gi[x] - yi - ri)) +
    sum(xsij*(aj - xj) + etaj*(xj - bj)) -
    sum(muj*yj)

lami    >= 0    := Lagrange multipliers wrt     gi[x] - yi <= ri
xsij    >= 0    := Lagrange multipliers wrt     aj <= xj
etaj    >= 0    := Lagrange multipliers wrt     xj <= bj
muj     >= 0    := Lagrange multipliers wrt     yi >= 0

L           = psi[x,lam] +
                sum(ci*yi - lami*(yi + ri) - mui*yi) +
                sum(alphaj * (aj - xj) + betaj * (xj - bj))
psi[x,lam]  = g0[x] + sum(lami * gi[x])

KKT conditions:

psi/dxj - xsij + etaj   =   0       (dL/dxj = 0)
ci + di*yi - lami - mui =   0       (dL/dyi = 0)

gi[x] - yi - ri  <=  0              (primal feasibility)
aj - xj                 <=  0       (primal feasibility)
xj - bj                 <=  0       (primal feasibility)

etaj                    >=  0       (dual feasibility)
xsij                    >=  0       (dual feasibility)
lami                    >=  0       (dual feasibility)
mui                     >=  0       (dual feasibility)

lami * (gi[x] - yi - ri)     =   0       (complementary slackness)
xsij * (aj - xj)                    =   0       (complementary slackness)
etaj * (xj - bj)                    =   0       (complementary slackness)
-mui*yi                             =   0       (complementary slackness)


The zeros in the right hand sides of the complementary slackness conditions are replaced by
a "small" negative parameter epsi > 0.

Slack variables si are introduced for the constraints.

RELAXED KKT conditions:

dL/dxj, dL/dyi are zero

gi[x] - yi - ri +si  <=  0       (primal feasibility)
aj - xj                     <   0       (primal feasibility)
xj - bj                     <   0       (primal feasibility)

etaj                    >  0       (dual feasibility)
xsij                    >  0       (dual feasibility)
lami                    >  0       (dual feasibility)
mui                     >  0       (dual feasibility)
yi                      >  0
si                      >  0

lami * (gi[x] - yi - ri) - epsi     =   0       (complementary slackness)
xsij * (aj - xj) - epsi                    =   0       (complementary slackness)
etaj * (xj - bj) - epsi                    =   0       (complementary slackness)
-mui*yi - epsi                             =   0       (complementary slackness)
lami*si - epsi                             =   0

Given a point w = (x,xsi,eta,lam,s,y,mu) which satisfies feasibility
one can apply Newton's method to obtain dw = (dx,dy,dlam,dxsi,deta,dmu,ds).
Here dxsi, deta and ds can be eliminated without severe computational effort.

Subsequently we are left with a reduced system in terms of dx and dlam
"""


class InteriorPointXY(InteriorPointX):
    def __init__(self, problem, **kwargs):
        super().__init__(problem, **kwargs)

        self.c = kwargs.get('c', 1000 * np.ones(self.m))

        """
        Svanberg's implementation uses w.x = (a + b)/2.
        I found w.x = x (that is use the old variable field as initial guess)
        to converge much faster.
        Note however that a < x < b must hold. For variables where this does not hold one should use
        w.x = (a + b)/2
        """

        self.w = StateY(
            *(var for var in self.w),
            np.ones(self.m),
            np.maximum(self.c / 2, 1))

        self.r = deepcopy(self.w)
        self.dw = deepcopy(self.w)

    def residual(self):
        """
        r(x)        = psi / dx - xsi + eta = dg/dx[x] obj + lam' * dg/dx[x] constraints - xsi + eta
        r(xsi)      = xsi * (x - a) - e
        r(eta)      = eta * (b - x) - e
        r(lam)      = g[x] + s - y
        r(s)        = lam * si - e

        r(y)        = c + d*y - lam - mu
        r(mu)       = mu*y - e
        """
        self.r.lam -= self.w.y  # relam
        self.r.y = self.c - self.w.mu - self.w.lam  # rey
        self.r.mu = self.w.mu * self.w.y - self.epsi  # remu
        return super().residual()

    def get_newton_direction(self):
        # Some calculations to omit repetitive calculations later on
        a = self.w.x - self.alpha
        b = self.beta - self.w.x
        g = self.g(self.w.x)
        dg = self.dg(self.w.x)
        ddg = self.ddg(self.w.x)

        # delta_lambda
        delta_lambda = g[1:] - self.w.y + self.epsi / self.w.lam
        delta_x = dg[0] + self.w.lam.dot(dg[1:]) - self.epsi / a + self.epsi / b
        delta_y = self.c - self.w.lam - self.epsi / self.w.y

        diag_lambda = self.w.s / self.w.lam  # s./lam
        diag_x = ddg[0] + self.w.lam.dot(ddg[1:]) + self.w.xsi / a + self.w.eta / b
        diag_y = self.w.mu / self.w.y

        diag_lambday = diag_lambda + 1 / diag_y
        delta_lambday = delta_lambda + delta_y / diag_y

        # #FIXME: the m > n needs to be checked
        # if self.m > self.n:
        #     # dldl = delta_lambda/diag_lambda
        #     Bx = -delta_x - (delta_lambday/diag_lambday).dot(dg[1:])
        #     Ax = diags(diag_x) + dg[1:].transpose().dot(diags(1/diag_lambday) * dg[1:])
        #     ax = - (self.a/diag_lambday).dot(dg[1:])
        #     az = self.w.zeta/self.w.z + - (self.a/diag_lambday).dot(self.a)
        #
        #     # solve for dx
        #     X = np.linalg.solve(np.block([[Ax, ax], [ax.transpose(), az]]),
        #                                     np.block([[Bx], [-delta_z + - (delta_lambday/diag_lambday).dot(self.a)]]))  # n x n
        #     self.dw[0][:] = X[:-1]
        #     self.dw[7][:] = X[-1]
        #     self.dw[3][:] = (delta_lambday + self.dw[0].dot(dg[1:]) + self.dw[7]*self.a)/diag_lambday
        #
        # else:
        dxdx = delta_x / diag_x
        Blam = delta_lambday - dxdx.dot(dg[1:].transpose())
        Alam = diags(diag_lambday) + np.einsum("ki,i,ji->kj", dg[1:], 1 / diag_x, dg[1:])  # calculate dx[lam]

        # solve for dlam
        self.dw.lam[:] = np.linalg.solve(Alam, Blam)
        self.dw.x = -dxdx - (self.dw.lam.dot(dg[1:])) / diag_x

        # get dxsi[dx], deta[dx] and ds[dlam]
        self.dw.xsi = -self.w.xsi + self.epsi / a - (self.w.xsi * self.dw.x) / a
        self.dw.eta = -self.w.eta + self.epsi / b + (self.w.eta * self.dw.x) / b
        self.dw.s = -self.w.s + self.epsi / self.w.lam - (self.w.s * self.dw.lam) / self.w.lam
        self.dw.y = (self.dw.lam - delta_y) / diag_y
        self.dw.mu = -self.w.mu + self.epsi / self.w.y - (self.w.mu * self.dw.y) / self.w.y
