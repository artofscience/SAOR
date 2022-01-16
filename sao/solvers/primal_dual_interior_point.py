from abc import ABC
from copy import deepcopy
from dataclasses import dataclass, fields
from numba import njit
import numpy as np
from scipy.sparse import diags


@dataclass
class State(object):
    """A dataclass carrying the state for the IPOpt solvers.

    This dataclass contains the often used variables throughout the IP solvers.
    Additionally, some basic functionality is provided, such as addition
    and multiplication. For addition, two states can be added where the values
    of the corresponding fields are summed together. For multiplication, it is
    assumed to be multiplied with a single scalar value, scaling_strategies all fields
    within the state object.
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
class StateZ(StateY):
    """An extended data class adding the ``z`` and ``zeta`` variables."""
    z: np.array
    zeta: np.array


class Pdip(ABC):
    def __init__(self, problem, **kwargs):
        self.problem = problem

    r: State = NotImplemented
    w: State = NotImplemented
    dw: State = NotImplemented
    wold: State = NotImplemented

    def get_point(self):
        return self.w.x - self.problem.x_min, \
               self.problem.x_max - self.w.x, \
               self.problem.g(self.w.x), \
               self.problem.dg(self.w.x), \
               self.problem.ddg(self.w.x)

    def residual(self, epsi):
        ...

    def get_newton_direction(self, epsi):
        ...

    def get_step_size(self, alphab=-1.01):
        step_x = [np.max(alphab * dw / w) for w, dw in zip(tuple(self.w)[1:], tuple(self.dw)[1:])]
        step_alpha = np.max(alphab * self.dw.x / (self.w.x - self.problem.x_min))
        step_beta = np.max(-alphab * self.dw.x / (self.problem.x_max - self.w.x))
        return 1.0 / max(np.asarray([1.0, np.max(step_x), step_alpha, step_beta]))


class Pdipx(Pdip):
    def __init__(self, problem, x0):
        super().__init__(problem)

        self.w = State(
            x0,
            np.maximum(1 / (x0 - self.problem.x_min), 1),
            np.maximum(1 / (self.problem.x_max - x0), 1),
            np.ones(self.problem.m),
            np.ones(self.problem.m)
        )

        self.r = deepcopy(self.w)
        self.dw = deepcopy(self.w)
        self.wold = deepcopy(self.w)

    def residual(self, epsi):
        """Updates the residual and return its norm and maximum.

        r(x)        = psi / dx - xsi + eta = dg/dx[x] obj + lam' * dg/dx[x] constraints - xsi + eta
        r(xsi)      = xsi * (x - a) - e
        r(eta)      = eta * (b - x) - e
        r(lam)      = gi[x] - ri + si
        r(s)        = lam * si - e
        """
        dg = self.problem.dg(self.w.x)
        self.r.x = dg[0] + self.w.lam.dot(dg[1:]) - self.w.xsi + self.w.eta
        self.r.xsi = self.w.xsi * (self.w.x - self.problem.x_min) - epsi
        self.r.eta = self.w.eta * (self.problem.x_max - self.w.x) - epsi
        self.r.lam = self.problem.g(self.w.x)[1:] + self.w.s
        self.r.s = self.w.lam * self.w.s - epsi
        return self.r.norm(), self.r.max()

    def get_newton_direction(self, epsi):
        a, b, g, dg, ddg = self.get_point()

        # delta_lambda
        delta_lambda = g[1:] + epsi / self.w.lam
        delta_x = dg[0] + self.w.lam.dot(dg[1:]) - epsi / a + epsi / b

        diag_lambda = self.w.s / self.w.lam  # s./lam
        diag_x = ddg[0] + self.w.lam.dot(ddg[1:]) + self.w.xsi / a + self.w.eta / b

        if self.problem.m > self.problem.n:
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
        self.dw.xsi = -self.w.xsi + epsi / a - (self.w.xsi * self.dw.x) / a
        self.dw.eta = -self.w.eta + epsi / b + (self.w.eta * self.dw.x) / b
        self.dw.s = -self.w.s + epsi / self.w.lam - (self.w.s * self.dw.lam) / self.w.lam


class Pdipxy(Pdipx):
    def __init__(self, problem, x0, c=1000):
        super().__init__(problem, x0)

        self.c = c * np.ones(problem.m)

        """
        Svanberg's implementation uses w.x = (a + b)/2.
        I found w.x = x (that is use the old variable field as initial guess)
        to converge much faster.
        Note however that a < x < b must hold. For variables where this does not hold one should use
        w.x = (a + b)/2
        """

        self.w = StateY(
            *(var for var in self.w),
            np.ones(problem.m),
            np.maximum(self.c / 2, 1))

        self.r = deepcopy(self.w)
        self.dw = deepcopy(self.w)

    def residual(self, epsi):
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
        self.r.mu = self.w.mu * self.w.y - epsi  # remu
        return super().residual(epsi)

    def get_newton_direction(self, epsi):
        a, b, g, dg, ddg = self.get_point()

        # delta_lambda
        delta_lambda = g[1:] - self.w.y + epsi / self.w.lam
        delta_x = dg[0] + self.w.lam.dot(dg[1:]) - epsi / a + epsi / b
        delta_y = self.c - self.w.lam - epsi / self.w.y

        diag_lambda = self.w.s / self.w.lam  # s./lam
        diag_x = ddg[0] + self.w.lam.dot(ddg[1:]) + self.w.xsi / a + self.w.eta / b
        diag_y = self.w.mu / self.w.y

        diag_lambday = diag_lambda + 1 / diag_y
        delta_lambday = delta_lambda + delta_y / diag_y

        dxdx = delta_x / diag_x
        Blam = delta_lambday - dxdx.dot(dg[1:].transpose())
        Alam = diags(diag_lambday) + np.einsum("ki,i,ji->kj", dg[1:], 1 / diag_x, dg[1:])  # calculate dx[lam]

        # solve for dlam
        self.dw.lam[:] = np.linalg.solve(Alam, Blam)
        self.dw.x = -dxdx - (self.dw.lam.dot(dg[1:])) / diag_x

        # get dxsi[dx], deta[dx] and ds[dlam]
        self.dw.xsi = -self.w.xsi + epsi / a - (self.w.xsi * self.dw.x) / a
        self.dw.eta = -self.w.eta + epsi / b + (self.w.eta * self.dw.x) / b
        self.dw.s = -self.w.s + epsi / self.w.lam - (self.w.s * self.dw.lam) / self.w.lam
        self.dw.y = (self.dw.lam - delta_y) / diag_y
        self.dw.mu = -self.w.mu + epsi / self.w.y - (self.w.mu * self.dw.y) / self.w.y


class Pdipxyz(Pdipxy):
    def __init__(self, problem, x0, a0=1.):
        super().__init__(problem, x0)

        self.a0 = a0
        self.d = np.zeros(problem.m)
        self.a = np.zeros(problem.m)

        """
        Svanberg's implementation uses w.x = (a + b)/2.
        I found w.x = x (that is use the old variable field as initial guess)
        to converge much faster.
        Note however that a < x < b must hold. For variables where this does not hold one should use
        w.x = (a + b)/2
        """

        self.w = StateZ(
            *(var for var in self.w),
            np.ones(1),
            np.ones(1))

        self.r = deepcopy(self.w)
        self.dw = deepcopy(self.w)

    def residual(self, epsi):
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
        self.r.zeta = self.w.z * self.w.zeta - epsi  # rezet
        return super().residual(epsi)

    def get_newton_direction(self, epsi):
        a, b, g, dg, ddg = self.get_point()

        # delta_lambda
        delta_lambda = g[1:] - self.w.y + epsi / self.w.lam - self.a * self.w.z
        delta_x = dg[0] + self.w.lam.dot(dg[1:]) - epsi / a + epsi / b
        delta_y = self.c - self.w.lam - epsi / self.w.y
        delta_z = self.a0 - np.dot(self.w.lam, self.a) - epsi / self.w.z

        diag_lambda = self.w.s / self.w.lam  # s./lam
        diag_x = ddg[0] + self.w.lam.dot(ddg[1:]) + self.w.xsi / a + self.w.eta / b
        diag_y = self.w.mu / self.w.y

        diag_lambday = diag_lambda + 1 / diag_y
        delta_lambday = delta_lambda + delta_y / diag_y

        dxdx = delta_x / diag_x
        zzeta = self.w.z / self.w.zeta
        Blam = delta_lambday - dxdx.dot(dg[1:].transpose()) + zzeta * self.a * delta_z
        Alam = diags(diag_lambday) + np.einsum("ki,i,ji->kj", dg[1:], 1 / diag_x, dg[1:]) + zzeta * self.a * self.a.T

        # solve for dlam
        self.dw.lam[:] = np.linalg.solve(Alam, Blam)
        self.dw.x = -dxdx - (self.dw.lam.dot(dg[1:])) / diag_x
        self.dw.z = zzeta * (np.dot(self.a, self.dw.lam) - delta_z)

        # get dxsi[dx], deta[dx] and ds[dlam]
        self.dw.xsi = -self.w.xsi + epsi / a - (self.w.xsi * self.dw.x) / a
        self.dw.eta = -self.w.eta + epsi / b + (self.w.eta * self.dw.x) / b
        self.dw.s = -self.w.s + epsi / self.w.lam - (self.w.s * self.dw.lam) / self.w.lam
        self.dw.y = (self.dw.lam - delta_y) / diag_y
        self.dw.mu = -self.w.mu + epsi / self.w.y - (self.w.mu * self.dw.y) / self.w.y
        self.dw.zeta = -1 / zzeta * self.dw.z - self.w.zeta + epsi / self.w.z


def pdip(problem, x0=None, variables=Pdipxyz, epsimin=1e-9, max_outer_iter=100,
         max_lines_iter=20, max_inner_iter=20, epsifac=0.9, epsired=0.1):
    if x0 is None:
        x0 = 0.5 * (problem.x_min + problem.x_max)

    iter = 0
    epsi = 1
    counter = 0

    state = variables(problem, x0)

    while iter < max_outer_iter and epsi > epsimin:
        iter += 1

        rnorm, rmax = state.residual(epsi)

        inner_iter = 0
        while inner_iter < max_inner_iter and rmax > epsifac * epsi:
            inner_iter = inner_iter + 1
            counter += 1

            state.get_newton_direction(epsi)
            state.wold = deepcopy(state.w)
            step = state.get_step_size()

            lines_iter = 0
            rnew = 2 * rnorm
            while lines_iter < max_lines_iter and rnew > rnorm:
                lines_iter += 1

                state.w = state.wold + step * state.dw
                rnew, rmax = state.residual(epsi)

                step /= 2
            rnorm = rnew
            step *= 2
        epsi *= epsired
    return state.w.x, counter
