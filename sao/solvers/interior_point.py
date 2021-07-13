from .primal_dual import PrimalDual
import numpy as np
from abc import ABC, abstractmethod
from copy import deepcopy
from scipy.sparse import diags
from dataclasses import dataclass, fields


@dataclass
class State(object):
    """A dataclass carrying the state for the IPOpt solvers.

    This dataclass contains the often used variables throughout the IP solvers.
    Additionally, some basic functionality is provided, such as addition
    and multiplication. For addition, two states can be added where the values
    of the corresponding fields are summed together. For multiplication, it is
    assumed to be multiplied with a single scalar value, scaling_strategies all fields
    within the state object.

    TODO: document the variables and what they represent
    """
    x: np.array
    xsi: float
    eta: float
    lam: np.array
    s: np.array

    def __iter__(self):
        """Iterator over all values of each field present in the state."""
        for field in fields(self):
            yield getattr(self, field.name)

    def __add__(self, other):
        """Addition of two states, where each field is summed together."""
        return self.__class__(*(s + o for s, o in zip(self, other)))

    def __mul__(self, other):
        """Multiplication with a scalar value to scale all fields."""
        return self.__class__(*(value * other for value in self))

    def __rmul__(self, other):
        """Right multiplication with a scalar, see ``__mul__``."""
        return self.__mul__(other)

    def norm(self):
        """Return the norm of all stacked variables."""
        return np.linalg.norm(np.hstack(tuple(self)))

    def max(self):
        """Return the abs maximum of all stacked variables in self."""
        return np.max(np.abs(np.hstack(tuple(self))))


@dataclass
class StateY(State):
    """An extended data class adding the ``y`` and ``mu`` variables."""
    y: np.array
    mu: float


@dataclass
class StateYZ(StateY):
    """An extended data class adding the ``z`` and ``zeta`` variables."""
    z: np.array
    zeta: np.array


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
            dldl = delta_lambda/diag_lambda
            B = -delta_x - dldl.dot(dg[1:])
            A = diags(diag_x) + dg[1:].transpose().dot(diags(1/diag_lambda) * dg[1:])

            # solve for dx
            self.dw.x[:] = np.linalg.solve(A, B)  # n x n
            self.dw.lam[:] = dg[1:].dot(self.dw.x)/diag_lambda + dldl  # calculate dlam[dx]

        else:
            dxdx = delta_x / diag_x
            B = delta_lambda - dxdx.dot(dg[1:].transpose())
            A = diags(diag_lambda) + np.einsum("ki,i,ji->kj", dg[1:], 1/diag_x, dg[1:])

            # solve for dlam
            self.dw.lam = np.linalg.solve(A, B)  # m x m
            self.dw.x = -dxdx - (self.dw.lam.dot(dg[1:])) / diag_x

        # get dxsi[dx], deta[dx] and ds[dlam]
        self.dw.xsi = -self.w.xsi + self.epsi / a - (self.w.xsi * self.dw.x) / a
        self.dw.eta = -self.w.eta + self.epsi / b + (self.w.eta * self.dw.x) / b
        self.dw.s = -self.w.s + self.epsi / self.w.lam - (self.w.s * self.dw.lam) / self.w.lam

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
        Alam = diags(diag_lambday) + np.einsum("ki,i,ji->kj", dg[1:], 1/diag_x, dg[1:])  # calculate dx[lam]

        # solve for dlam
        self.dw.lam[:] = np.linalg.solve(Alam, Blam)
        self.dw.x = -dxdx - (self.dw.lam.dot(dg[1:])) / diag_x

        # get dxsi[dx], deta[dx] and ds[dlam]
        self.dw.xsi = -self.w.xsi + self.epsi / a - (self.w.xsi * self.dw.x) / a
        self.dw.eta = -self.w.eta + self.epsi / b + (self.w.eta * self.dw.x) / b
        self.dw.s = -self.w.s + self.epsi / self.w.lam - (self.w.s * self.dw.lam) / self.w.lam
        self.dw.y = (self.dw.lam - delta_y) / diag_y
        self.dw.mu = -self.w.mu + self.epsi / self.w.y - (self.w.mu * self.dw.y) / self.w.y

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
        self.a0 = kwargs.get('a0', 1.)
        self.d = kwargs.get('d', np.zeros(self.m))
        self.a = kwargs.get('a', np.zeros(self.m))

        """
        Svanberg's implementation uses w.x = (a + b)/2.
        I found w.x = x (that is use the old variable field as initial guess)
        to converge much faster.
        Note however that a < x < b must hold. For variables where this does not hold one should use
        w.x = (a + b)/2
        """

        self.w = StateYZ(
                *(var for var in self.w),
                np.ones(1),
                np.ones(1))

        self.r = deepcopy(self.w)
        self.dw = deepcopy(self.w)

    def residual(self):
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
        self.r.lam -= self.w.z * self.a  # relam
        self.r.y += self.d * self.w.y  # rey
        self.r.z = self.a0 - self.w.zeta - self.w.lam.dot(self.a)  # rez
        self.r.zeta = self.w.z * self.w.zeta - self.epsi  # rezet
        return super().residual()

    def get_newton_direction(self):
        # Some calculations to omit repetitive calculations later on
        a = self.w.x - self.alpha
        b = self.beta - self.w.x
        g = self.g(self.w.x)
        dg = self.dg(self.w.x)
        ddg = self.ddg(self.w.x)

        # delta_lambda
        delta_lambda = g[1:] - self.a * self.w.z - self.w.y + self.epsi / self.w.lam
        delta_x = dg[0] + self.w.lam.dot(dg[1:]) - self.epsi / a + self.epsi / b
        delta_y = self.c + self.d * self.w.y - self.w.lam - self.epsi / self.w.y
        delta_z = self.a0 - self.w.lam.dot(self.a) - self.epsi / self.w.z

        diag_lambda = self.w.s / self.w.lam  # s./lam
        diag_x = ddg[0] + self.w.lam.dot(ddg[1:]) + self.w.xsi / a + self.w.eta / b
        diag_y = self.d + self.w.mu / self.w.y

        diag_lambday = diag_lambda + 1 / diag_y
        delta_lambday = delta_lambda + delta_y / diag_y

        # FIXME: the m > n needs to be checked
        if self.m > self.n:
            # dldl = delta_lambda/diag_lambda
            Bx = -delta_x - (delta_lambday / diag_lambday).dot(dg[1:])
            Ax = diags(diag_x) + dg[1:].transpose().dot(diags(1 / diag_lambday) * dg[1:])
            ax = - (self.a / diag_lambday).dot(dg[1:])
            az = self.w.zeta / self.w.z + - (self.a / diag_lambday).dot(self.a)

            # solve for dx
            X = np.linalg.solve(np.block([[Ax, ax[np.newaxis, :].transpose()], [ax[np.newaxis,:], az]]),
                                np.block([[Bx[:,np.newaxis]], [-delta_z - (delta_lambday / diag_lambday).dot(self.a)]])).flatten() # n x n
            self.dw.x = X[:-1]
            self.dw.z = X[-1]
            self.dw.lam = (delta_lambday + dg[1:].dot(self.dw.x) + self.dw.z * self.a) / diag_lambday

        else:
            dxdx = delta_x / diag_x
            Blam = delta_lambday - dxdx.dot(dg[1:].transpose())
            Alam = diags(diag_lambday) + np.einsum("ki,i,ji->kj", dg[1:], 1/diag_x, dg[1:])  # calculate dx[lam]

            # solve for dlam
            X = np.linalg.solve(np.block([[Alam, self.a], [self.a.transpose(), -self.w.zeta / self.w.z]]), np.block([[Blam], [delta_z]]))
            self.dw.lam[:] = X[:-1]  # m x m # here I cannot remove the [:], why?
            self.dw.z = X[-1]
            self.dw.x = -dxdx - (self.dw.lam.dot(dg[1:])) / diag_x

        # get dxsi[dx], deta[dx] and ds[dlam]
        self.dw.xsi = -self.w.xsi + self.epsi / a - (self.w.xsi * self.dw.x) / a
        self.dw.eta = -self.w.eta + self.epsi / b + (self.w.eta * self.dw.x) / b
        self.dw.s = -self.w.s + self.epsi / self.w.lam - (self.w.s * self.dw.lam) / self.w.lam
        self.dw.y = (self.dw.lam - delta_y) / diag_y
        self.dw.mu = -self.w.mu + self.epsi / self.w.y - (self.w.mu * self.dw.y) / self.w.y
        self.dw.zeta = -self.w.zeta + self.epsi / self.w.z - (self.w.zeta * self.dw.z) / self.w.z
