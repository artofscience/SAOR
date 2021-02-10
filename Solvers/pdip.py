import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve
from copy import deepcopy

"""
This file is written by:
S. Koppen
s.koppen@tudelft.nl

T

This is a primal-dual interior-point approach (without slack variables)
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
class pdip:
    """
    Primal-dual interior point method.
    Construction provides the problem object which (at least) contains:
    g[x] (m+1 x 1)          Responses
    dg[x] (n x m+1)         Sensitivities
    ddg[x] (n x m+1)        (optionally 2nd order diagonal sensitivities)
    r (m+1 x 1)             zero order terms

    In addition it provides the current design point (x) and bounds (a and b)
    """

    def __init__(self, problem, x, a, b):

        self.x = x
        self.a = a
        self.b = b
        self.problem = problem
        self.n = self.problem.n
        self.m = self.problem.m

        # initialization of residual vectors
        self.r = [np.zeros(self.n), np.zeros(self.m), np.zeros(self.n), np.zeros(self.n), np.zeros(self.m)]
        self.w = [x, np.ones(self.m), np.max(1 / (x - self.a), 1), np.max(1 / (self.a - x), 1), np.ones(self.m)]
        self.dw = deepcopy(self.r)
        self.wold = deepcopy(self.w)

        self.iterout = 0
        self.iterin = 0
        self.itera = 0
        self.iteramax = 50
        self.iterinmax = 100

        self.step = 0

        # initialization of relaxed epsi parameter
        self.epsi = 1
        self.epsifac = 0.9
        self.epsimin = 1e-6
        self.ab = -1.01

    def update(self):

        # iterate until convergence
        while self.epsi > self.epsimin:
            self.get_residual()
            rnorm = np.linalg.norm(np.array([np.linalg.norm(i) for i in self.r.a]))
            rmax = np.max(np.array([np.max(i) for i in self.r.a]))

            self.iterin = 0
            while rmax > self.epsifac and self.iterin < self.iterinmax:
                self.iterin += 1
                self.iterout += 1

                # get the Newton direction
                self.get_newton_direction()

                for i, j in self.w:
                    self.wold[i][:] = j

                self.itera = 0
                rnew = 2*rnorm

                # Line search in the Newton direction dw
                while rnew > rnorm and self.itera < self.iteramax:
                    self.itera += 1

                    # calculate step size
                    self.step = 1/np.max(self.ab * np.array([self.dw.lam/self.w.lam,
                                                        self.dw.s/self.w.s,
                                                        self.dw.xsi/self.w.xsi,
                                                        self.dw.eta/self.w.eta,
                                                        self.dw.x/(self.w.x - self.a),
                                                        self.dw.x/(self.b - self.w.x)]))

                    # set a step in the Newton direction w^(l+1) = w^(l) + step^(l) * dw
                    for i,j in self.w:
                        j[:] = self.wold[i] + self.step * self.dw[i]

                    self.get_residual()
                    rnew = np.linalg.norm(np.array([np.linalg.norm(i) for i in self.r]))
                    self.step *= 0.5

                rnorm = 1.0 * rnew
                rmax = np.max(np.array([np.max(i) for i in self.r]))
                self.step *= 2

            self.epsi *= 0.5

        # end
    def get_residual(self):
        """
        r(x)        = psi / dx - xsi + eta
        r(xsi)      = xsi * (x - a) - e
        r(eta)      = eta * (b - x) - e
        r(lam)      = gi[x] - ri + si
        r(s)        = lam * si - e
        """

        self.r[0] = self.problem.dg[0] + np.dot(self.problem.dg(self.x)[1:], self.w[1]) - self.w[2] + self.w[3]
        self.r[1] = self.problem.g(self.x)[1:] - self.problem.r[1:] + self.w[4]
        self.r[2] = np.dot(self.w[2], self.w[0] - self.a) - self.epsi
        self.r[3] = np.dot(self.w[3], self.b - self.w[0]) - self.epsi
        self.r[4] = np.dot(self.w[1], self.w[4]) - self.epsi

    def get_newton_direction(self):
        a = self.w[0] - self.a
        b = self.b - self.w[0]
        g = self.problem.g(self.w[0])
        dg = self.problem.dg(self.w[0])
        ddg = self.problem.ddg(self.w[0])

        # delta_lambda
        delta_lambda = g[1:] - self.problem.r[1:] + self.epsi/self.w[1]
        delta_x = dg[0] + np.dot(dg[1:], self.w[1]) - self.epsi/a + self.epsi/b

        diag_lambda = self.w[4]/self.w[1]
        diag_x  = ddg[0] + np.dot(ddg[1:], self.w[1]) - self.w[2]/a + self.w[3]/b

        # FIXME: implement dense solvers and CG
        if self.problem.m > self.problem.n:
            B = -delta_x - np.transpose(dg[1:]) * (delta_lambda/diag_lambda)
            A = diags(diag_x) + np.transpose(g[1:]) * (diags(1/diag_lambda) * dg[1:])

            # sovle for dx
            self.dw[0] = spsolve(A, B)
            self.dw[1] = (dg[1:] * self.dw[0])/diag_lambda - delta_lambda/diag_lambda

        else:
            B = delta_lambda - (dg[1:] * delta_x/diag_x)
            A = diags(diag_lambda) + dg[1:] * (diags(1/diag_x) * np.transpose(dg[1:]))

            # solve for dlam
            self.dw[1] = spsolve(A, B) # m x m
            self.dw[0] = -delta_x/diag_x - (np.transpose(dg[1:]) * self.dw[1])/diag_x

        # get dxsi[dx], deta[dx] and ds[dlam]
        self.dw[2] = -self.w[2] + self.epsi/a - np.dot(self.w[2], self.dw[0])/a
        self.dw[3] = -self.w[3] + self.epsi/b - np.dot(self.w[3], self.dw[0])/b
        self.dw[4] = -self.w[4] + self.epsi/self.w[1] - np.dot(self.w[4], self.dw[1])/self.w[1]

