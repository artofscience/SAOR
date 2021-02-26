import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve
from copy import deepcopy
from Solvers.interior_point import InteriorPoint

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


# this class should be a child of an abstract solver class
class InteriorPointBasis(InteriorPoint):
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
        #FIXME: implement correct initialization
        self.w = [self.x,  # x
                  np.maximum(1/(self.x - self.alpha), 1),  # xsi
                  np.maximum(1/(self.beta - self.x), 1),  # eta
                  np.ones(self.m),  # lam
                  np.ones(self.m)]  # s

        self.r = [np.zeros(self.n),
                  np.zeros(self.n),
                  np.zeros(self.n),
                  np.zeros(self.m),
                  np.zeros(self.m)]

        self.dw = deepcopy(self.r)
        self.wold = deepcopy(self.w)

    def get_step_size(self):
        temp = [self.alphab * self.dw[i+1] / w for i, w in enumerate(self.w[1:])]
        temp.append(self.alphab * self.dw[0] / (self.w[0] - self.alpha))
        temp.append(self.alphab * self.dw[0] / (self.beta - self.w[0]))
        temp.append(np.array([1]))
        self.step = 1 / max([max(i) for i in temp])

    def get_residual(self):
        """
        r(x)        = psi / dx - xsi + eta = dg/dx[x] obj + lam' * dg/dx[x] constraints - xsi + eta
        r(xsi)      = xsi * (x - a) - e
        r(eta)      = eta * (b - x) - e
        r(lam)      = gi[x] - ri + si
        r(s)        = lam * si - e
        """

        self.r[0][:] = self.dg(self.w[0])[0] + self.w[3].dot(self.dg(self.w[0])[1:]) - self.w[1] + self.w[2]
        self.r[1][:] = self.w[1] * (self.w[0] - self.alpha) - self.epsi
        self.r[2][:] = self.w[2] * (self.beta - self.w[0]) - self.epsi
        self.r[3][:] = self.g(self.w[0])[1:] - self.zo[1:] + self.w[4]
        self.r[4][:] = self.w[3] * self.w[4] - self.epsi

    def get_newton_direction(self):
        # Some calculations to omit repetitive calculations later on
        a = self.w[0] - self.alpha
        b = self.beta - self.w[0]
        g = self.g(self.w[0])
        dg = self.dg(self.w[0])
        ddg = self.ddg(self.w[0])

        # delta_lambda
        delta_lambda = g[1:] - self.zo[1:] + self.epsi/self.w[3]
        delta_x = dg[0] + self.w[3].dot(dg[1:]) - self.epsi/a + self.epsi/b

        diag_lambda = self.w[4]/self.w[3]  # s./lam
        diag_x = ddg[0] + self.w[3].dot(ddg[1:]) + self.w[1]/a + self.w[2]/b

        # FIXME: implement dense solvers and CG
        if self.m > self.n:
            dldl = delta_lambda/diag_lambda
            B = -delta_x - np.transpose(dg[1:]) * dldl
            A = diags(diag_x) + np.transpose(g[1:]) * (diags(1/diag_lambda) * dg[1:])

            # solve for dx
            self.dw[0] = spsolve(A, B)  # n x n
            self.dw[1] = (dg[1:] * self.dw[0])/diag_lambda - dldl  # calculate dlam[dx]

        else:
            dxdx = delta_x/diag_x
            B = delta_lambda - dxdx.dot(dg[1:].transpose())
            A = diags(diag_lambda) + dg[1:].dot(diags(1/diag_x) * dg[1:].transpose())  # calculate dx[lam]

            # solve for dlam
            self.dw[3][:] = np.linalg.solve(A, B)  # m x m
            self.dw[0][:] = -dxdx - (self.dw[3].dot(dg[1:]))/diag_x

        # get dxsi[dx], deta[dx] and ds[dlam]
        self.dw[1][:] = -self.w[1] + self.epsi/a - (self.w[1] * self.dw[0])/a
        self.dw[2][:] = -self.w[2] + self.epsi/b + (self.w[2] * self.dw[0])/b
        self.dw[4][:] = -self.w[4] + self.epsi/self.w[3] - (self.w[4] * self.dw[3])/self.w[3]
