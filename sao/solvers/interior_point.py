from .primal_dual import PrimalDual
import numpy as np
from abc import ABC, abstractmethod
from copy import deepcopy
from scipy.sparse import diags


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
        super().__init__(problem, **kwargs)

        self.epsimin = kwargs.get('epsimin', 1e-4)
        self.iteramax = kwargs.get('iteramax', 50)
        self.iterinmax = kwargs.get('iterinmax', 20)
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

    def get_step_size(self):
        temp = [self.alphab * self.dw[1:][i] / a for i, a in enumerate(self.w[1:])]
        temp.append(self.alphab * self.dw[0] / (self.w[0] - self.alpha))
        temp.append(-self.alphab * self.dw[0] / (self.beta - self.w[0]))
        temp.append(np.array([1]))
        self.step = 1 / np.max(np.hstack(temp))

    def update(self):

        # iterate until convergence
        self.itera = 0
        while self.epsi > self.epsimin:
            self.iter += 1

            # Calculate the initial residual, its norm and maximum value
            # This indicates how far we are from the global optimum for THIS epsi
            self.get_residual()
            rnorm = np.linalg.norm(np.hstack(self.r))
            rmax = np.max(np.abs(np.hstack(self.r)))

            ittt = 0
            while rmax > self.epsifac * self.epsi and ittt < self.iterinmax:
                ittt = ittt + 1
                self.itera += 1
                self.iterin += 1

                """
                Get the Newton direction
                This basically builds dw and includes a solve
                """
                self.get_newton_direction()
                self.get_step_size()

                # Set w_old = w
                for count, value in enumerate(self.w):
                    self.wold[count] = value

                # Initialize the counter for the line search
                rnew = 2 * rnorm

                # Line search in the Newton direction dw
                itto = 0
                while rnew > rnorm and itto < self.iteramax:
                    itto = itto + 1

                    # set a step in the Newton direction w^(l+1) = w^(l) + step^(l) * dw
                    for count, value in enumerate(self.dw):
                        self.w[count] = self.wold[count] + self.step * value

                    self.get_residual()
                    rnew = np.linalg.norm(np.hstack(self.r))
                    self.step *= 0.5

                rnorm = 1.0 * rnew
                rmax = np.max(np.abs(np.hstack(self.r)))
                self.step *= 2

            self.epsi *= self.epsired

            # print("iter: %2d" % self.iter, ", ",
            #       "solves: %2d" % itera, ", ",
            #       "obj: %.2e" % self.g(self.w[0])[0], ", ",
            #       "x:", ["{:+.2f}".format(i) for i in self.w[0][1:3]], ", ",
            #       "lam:", ["{:+.2f}".format(i) for i in self.w[3]], ", ",
            #       "|kkt|: %.1e" % (np.linalg.norm(self.dg(self.w[0])[0] + self.w[3].dot(self.dg(self.w[0])[1:]))))

        return self.w[0]

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
        # FIXME: implement correct initialization
        self.w = [self.x0,  # x
                  np.maximum(1 / (self.x0 - self.alpha), 1),  # xsi
                  np.maximum(1 / (self.beta - self.x0), 1),  # eta
                  np.ones(self.m),  # lam
                  np.ones(self.m)]  # s

        self.r = deepcopy(self.w)
        self.dw = deepcopy(self.w)
        self.wold = deepcopy(self.w)

    def get_residual(self):
        """
        r(x)        = psi / dx - xsi + eta = dg/dx[x] obj + lam' * dg/dx[x] constraints - xsi + eta
        r(xsi)      = xsi * (x - a) - e
        r(eta)      = eta * (b - x) - e
        r(lam)      = gi[x] - ri + si
        r(s)        = lam * si - e
        """
        dg = self.dg(self.w[0])
        self.r[0] = dg[0] + self.w[3].dot(dg[1:]) - self.w[1] + self.w[2]
        self.r[1] = self.w[1] * (self.w[0] - self.alpha) - self.epsi
        self.r[2] = self.w[2] * (self.beta - self.w[0]) - self.epsi
        self.r[3] = self.g(self.w[0])[1:] + self.w[4]
        self.r[4] = self.w[3] * self.w[4] - self.epsi

    def get_newton_direction(self):
        # Some calculations to omit repetitive calculations later on
        a = self.w[0] - self.alpha
        b = self.beta - self.w[0]
        g = self.g(self.w[0])
        dg = self.dg(self.w[0])
        ddg = self.ddg(self.w[0])

        # delta_lambda
        delta_lambda = g[1:] + self.epsi / self.w[3]
        delta_x = dg[0] + self.w[3].dot(dg[1:]) - self.epsi / a + self.epsi / b

        diag_lambda = self.w[4] / self.w[3]  # s./lam
        diag_x = ddg[0] + self.w[3].dot(ddg[1:]) + self.w[1] / a + self.w[2] / b

        if self.m > self.n:
            dldl = delta_lambda/diag_lambda
            B = -delta_x - dldl.dot(dg[1:])
            A = diags(diag_x) + dg[1:].transpose().dot(diags(1/diag_lambda) * dg[1:])

            # solve for dx
            self.dw[0][:] = np.linalg.solve(A, B)  # n x n
            self.dw[3][:] = dg[1:].dot(self.dw[0])/diag_lambda + dldl  # calculate dlam[dx]

        else:
            dxdx = delta_x / diag_x
            B = delta_lambda - dxdx.dot(dg[1:].transpose())
            A = diags(diag_lambda) + np.einsum("ki,i,ji->kj", dg[1:], 1/diag_x, dg[1:])

            # solve for dlam
            self.dw[3] = np.linalg.solve(A, B)  # m x m
            self.dw[0] = -dxdx - (self.dw[3].dot(dg[1:])) / diag_x

        # get dxsi[dx], deta[dx] and ds[dlam]
        self.dw[1] = -self.w[1] + self.epsi / a - (self.w[1] * self.dw[0]) / a
        self.dw[2] = -self.w[2] + self.epsi / b + (self.w[2] * self.dw[0]) / b
        self.dw[4] = -self.w[4] + self.epsi / self.w[3] - (self.w[4] * self.dw[3]) / self.w[3]

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
        # FIXME: implement correct initialization
        self.w.append(np.ones(self.m))  # y
        self.w.append(np.maximum(self.c / 2, 1))  # mu

        self.r = deepcopy(self.w)
        self.dw = deepcopy(self.w)
        self.wold = deepcopy(self.w)

    def get_residual(self):
        """
        r(x)        = psi / dx - xsi + eta = dg/dx[x] obj + lam' * dg/dx[x] constraints - xsi + eta
        r(xsi)      = xsi * (x - a) - e
        r(eta)      = eta * (b - x) - e
        r(lam)      = g[x] + s - y
        r(s)        = lam * si - e

        r(y)        = c + d*y - lam - mu
        r(mu)       = mu*y - e
        """
        super().get_residual()
        self.r[3] -= self.w[5]  # relam
        self.r[5] = self.c - self.w[6] - self.w[3]  # rey
        self.r[6] = self.w[6] * self.w[5] - self.epsi  # remu

    def get_newton_direction(self):
        # Some calculations to omit repetitive calculations later on
        a = self.w[0] - self.alpha
        b = self.beta - self.w[0]
        g = self.g(self.w[0])
        dg = self.dg(self.w[0])
        ddg = self.ddg(self.w[0])

        # delta_lambda
        delta_lambda = g[1:] - self.w[5] + self.epsi / self.w[3]
        delta_x = dg[0] + self.w[3].dot(dg[1:]) - self.epsi / a + self.epsi / b
        delta_y = self.c - self.w[3] - self.epsi / self.w[5]

        diag_lambda = self.w[4] / self.w[3]  # s./lam
        diag_x = ddg[0] + self.w[3].dot(ddg[1:]) + self.w[1] / a + self.w[2] / b
        diag_y = self.w[6] / self.w[5]

        diag_lambday = diag_lambda + 1 / diag_y
        delta_lambday = delta_lambda + delta_y / diag_y

        # #FIXME: the m > n needs to be checked
        # if self.m > self.n:
        #     # dldl = delta_lambda/diag_lambda
        #     Bx = -delta_x - (delta_lambday/diag_lambday).dot(dg[1:])
        #     Ax = diags(diag_x) + dg[1:].transpose().dot(diags(1/diag_lambday) * dg[1:])
        #     ax = - (self.a/diag_lambday).dot(dg[1:])
        #     az = self.w[8]/self.w[7] + - (self.a/diag_lambday).dot(self.a)
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
        Alam = diags(diag_lambday) + np.einsum("ki,i,ji->kj", dg[1:], 1/diag_x, dg[1:])  # calculate dx[lam]

        # solve for dlam
        self.dw[3][:] = np.linalg.solve(Alam, Blam)
        self.dw[0] = -dxdx - (self.dw[3].dot(dg[1:])) / diag_x

        # get dxsi[dx], deta[dx] and ds[dlam]
        self.dw[1] = -self.w[1] + self.epsi / a - (self.w[1] * self.dw[0]) / a
        self.dw[2] = -self.w[2] + self.epsi / b + (self.w[2] * self.dw[0]) / b
        self.dw[4] = -self.w[4] + self.epsi / self.w[3] - (self.w[4] * self.dw[3]) / self.w[3]
        self.dw[5] = (self.dw[3] - delta_y) / diag_y
        self.dw[6] = -self.w[6] + self.epsi / self.w[5] - (self.w[6] * self.dw[5]) / self.w[5]

"""
This is an addition to pdip, which extends problem P

Problem P:

min     g0[x]                       (objective)
s.t.    gi[x] <= ri,    i = 1...m   (constraints)
        aj <= xj <= bj  i = 1...n   (bound constraints)

and includes artificial variables y (m x 1) and z (1).


Consider the enriched Part

Problem Part:

min     g0[x] + a0 * z + sum(ci * yi + 0.5 * di * yi^2) (objective)
s.t.    gi[x] - ai * z - yi <= ri,    i = 1...m         (constraints)
        aj <= xj <= bj  i = 1...n                       (bound constraints)
        yi >= 0                                         (bound constraints)
        z >= 0

Constant real numbers: 
a0 > 0
ai >= 0
ci >= 0
di >= 0
ci + di > 0
ai * ci > a0 for all i with ai > 0

This makes it easier for the user to formulate certain subclasses of problems,
but also ensures a global optimum for problems that are infeasible using only the natural optimization variables.
However, in that case yi > 0.

To make Part (almost) equal to P, let a0 = 1 and ai = 0 for all i > 0.
Then z = 0 in any optimal solution of Part.
Further, let di = 1 and ci = "large", so that variables yi become relatively expensive.
Typically, y = 0 in any optimal solution of Part, and the corresponding x is an optimal solution of P as well.

The user should avoid "extremely large" value for ci (e.g. 10^10). 
It is good practice to start with low values for ci (e.g. 10^3) 
and raise if not all yi* = 0. (star denotes optimal solution)


Lagrangian L:

L := g0[x] + a0*z + sum(ci*yi + 0.5*di*yi^2) + 
    sum(lami * (gi[x] - ai*z - yi - ri)) + 
    sum(xsij*(aj - xj) + etaj*(xj - bj)) -
    sum(muj*yj) - zeta*z

lami    >= 0    := Lagrange multipliers wrt     gi[x] - ai * z - yi <= ri
xsij    >= 0    := Lagrange multipliers wrt     aj <= xj
etaj    >= 0    := Lagrange multipliers wrt     xj <= bj
muj     >= 0    := Lagrange multipliers wrt     yi >= 0
zeta    >= 0    := Lagrange multiplier wrt      z >= 0

L           = psi[x,lam] + 
                sum(ci*yi + 0.5*di*yi^2 - lami*(ai*z + yi + ri) - mui*yi) + 
                sum(alphaj * (aj - xj) + betaj * (xj - bj)) + 
                (a0 - zeta)*z
psi[x,lam]  = g0[x] + sum(lami * gi[x])

KKT conditions:

psi/dxj - xsij + etaj   =   0       (dL/dxj = 0)
ci + di*yi - lami - mui =   0       (dL/dyi = 0)
a0 - zeta - lami*ai     =   0       (dL/dz = 0)

gi[x] - ai*z - yi - ri  <=  0       (primal feasibility)
aj - xj                 <=  0       (primal feasibility)
xj - bj                 <=  0       (primal feasibility)

etaj                    >=  0       (dual feasibility)
xsij                    >=  0       (dual feasibility)
lami                    >=  0       (dual feasibility)
zeta                    >=  0       (dual feasibility)
mui                     >=  0       (dual feasibility)

lami * (gi[x] - ai*z - yi - ri)     =   0       (complementary slackness)
xsij * (aj - xj)                    =   0       (complementary slackness)
etaj * (xj - bj)                    =   0       (complementary slackness)
-mui*yi                             =   0       (complementary slackness)
-zeta*z                             =   0       (complementary slackness)

The zeros in the right hand sides of the complementary slackness conditions are replaced by
a "small" negative parameter epsi > 0.

Slack variables si are introduced for the constraints.

RELAXED KKT conditions:

dL/dxj, dL/dyi and dL/dz are zero

gi[x] - ai*z - yi - ri +si  <=  0       (primal feasibility)
aj - xj                     <   0       (primal feasibility)
xj - bj                     <   0       (primal feasibility)

etaj                    >  0       (dual feasibility)
xsij                    >  0       (dual feasibility)
lami                    >  0       (dual feasibility)
zeta                    >  0       (dual feasibility)
mui                     >  0       (dual feasibility)
z                       >  0
yi                      >  0
si                      >  0

lami * (gi[x] - ai*z - yi - ri) - epsi     =   0       (complementary slackness)
xsij * (aj - xj) - epsi                    =   0       (complementary slackness)
etaj * (xj - bj) - epsi                    =   0       (complementary slackness)
-mui*yi - epsi                             =   0       (complementary slackness)
-zeta*z - epsi                             =   0       (complementary slackness)
lami*si - epsi                             =   0

Given a point w = (x,xsi,eta,lam,s,y,mu,z,zeta) which satisfies feasibility
one can apply Newton's method to obtain dw = (dx,dy,dz,dlam,dxsi,deta,dmu,dzeta,ds).
Here dxsi, deta and ds can be eliminated without severe computational effort.

Subsequently we are left with a reduced system in terms of dx and dlam
"""


class InteriorPointXYZ(InteriorPointXY):
    def __init__(self, problem, **kwargs):
        super().__init__(problem, **kwargs)

        # self.c = kwargs.get('c', 1000 * np.ones(self.m))
        self.a0 = kwargs.get('a0', 1)
        self.d = kwargs.get('d', np.zeros(self.m))
        self.a = kwargs.get('a', np.zeros(self.m))

        """
        Svanberg's implementation uses w.x = (a + b)/2.
        I found w.x = x (that is use the old variable field as initial guess)
        to converge much faster.
        Note however that a < x < b must hold. For variables where this does not hold one should use
        w.x = (a + b)/2
        """

        self.w.append(np.ones(1))  # z
        self.w.append(np.ones(1))  # zeta

        self.r = deepcopy(self.w)
        self.dw = deepcopy(self.w)
        self.wold = deepcopy(self.w)

    def get_residual(self):
        """
        r(x)        = psi / dx - xsi + eta = dg/dx[x] obj + lam' * dg/dx[x] constraints - xsi + eta
        r(xsi)      = xsi * (x - a) - e
        r(eta)      = eta * (b - x) - e
        r(lam)      = g[x] + s - z*a - y
        r(s)        = lam * si - e

        r(y)        = c + d*y - lam - mu
        r(mu)       = mu*y - e
        r(z)        = a0 - zeta - lam.a
        r(zeta)     = z*zeta - e
        """
        super().get_residual()
        self.r[3] -= self.w[7] * self.a  # relam
        self.r[5] += self.d * self.w[5]  # rey
        self.r[7] = self.a0 - self.w[8] - self.w[3].dot(self.a)  # rez
        self.r[8] = self.w[7] * self.w[8] - self.epsi  # rezet

    def get_newton_direction(self):
        # Some calculations to omit repetitive calculations later on
        a = self.w[0] - self.alpha
        b = self.beta - self.w[0]
        g = self.g(self.w[0])
        dg = self.dg(self.w[0])
        ddg = self.ddg(self.w[0])

        # delta_lambda
        delta_lambda = g[1:] - self.a * self.w[7] - self.w[5] + self.epsi / self.w[3]
        delta_x = dg[0] + self.w[3].dot(dg[1:]) - self.epsi / a + self.epsi / b
        delta_y = self.c + self.d * self.w[5] - self.w[3] - self.epsi / self.w[5]
        delta_z = self.a0 - self.w[3].dot(self.a) - self.epsi / self.w[7]

        diag_lambda = self.w[4] / self.w[3]  # s./lam
        diag_x = ddg[0] + self.w[3].dot(ddg[1:]) + self.w[1] / a + self.w[2] / b
        diag_y = self.d + self.w[6] / self.w[5]

        diag_lambday = diag_lambda + 1 / diag_y
        delta_lambday = delta_lambda + delta_y / diag_y

        # FIXME: the m > n needs to be checked
        if self.m > self.n:
            # dldl = delta_lambda/diag_lambda
            Bx = -delta_x - (delta_lambday / diag_lambday).dot(dg[1:])
            Ax = diags(diag_x) + dg[1:].transpose().dot(diags(1 / diag_lambday) * dg[1:])
            ax = - (self.a / diag_lambday).dot(dg[1:])
            az = self.w[8] / self.w[7] + - (self.a / diag_lambday).dot(self.a)

            # solve for dx
            X = np.linalg.solve(np.block([[Ax, ax[np.newaxis, :].transpose()], [ax[np.newaxis,:], az]]),
                                np.block([[Bx[:,np.newaxis]], [-delta_z - (delta_lambday / diag_lambday).dot(self.a)]])).flatten() # n x n
            self.dw[0] = X[:-1]
            self.dw[7] = X[-1]
            self.dw[3] = (delta_lambday + dg[1:].dot(self.dw[0]) + self.dw[7] * self.a) / diag_lambday

        else:
            dxdx = delta_x / diag_x
            Blam = delta_lambday - dxdx.dot(dg[1:].transpose())
            Alam = diags(diag_lambday) + dg[1:].dot(diags(1 / diag_x) * dg[1:].transpose())  # calculate dx[lam]

            # solve for dlam
            X = np.linalg.solve(np.block([[Alam, self.a], [self.a.transpose(), -self.w[8] / self.w[7]]]), np.block([[Blam], [delta_z]]))
            self.dw[3][:] = X[:-1]  # m x m # here I cannot remove the [:], why?
            self.dw[7] = X[-1]
            self.dw[0] = -dxdx - (self.dw[3].dot(dg[1:])) / diag_x

        # get dxsi[dx], deta[dx] and ds[dlam]
        self.dw[1] = -self.w[1] + self.epsi / a - (self.w[1] * self.dw[0]) / a
        self.dw[2] = -self.w[2] + self.epsi / b + (self.w[2] * self.dw[0]) / b
        self.dw[4] = -self.w[4] + self.epsi / self.w[3] - (self.w[4] * self.dw[3]) / self.w[3]
        self.dw[5] = (self.dw[3] - delta_y) / diag_y
        self.dw[6] = -self.w[6] + self.epsi / self.w[5] - (self.w[6] * self.dw[5]) / self.w[5]
        self.dw[8] = -self.w[8] + self.epsi / self.w[7] - (self.w[8] * self.dw[7]) / self.w[7]
