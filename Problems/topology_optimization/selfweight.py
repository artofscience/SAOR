from .mbbbeam import MBBBeam
import numpy as np


class Selfweight(MBBBeam):
    def __init__(self, nelx, nely, volfrac, penal, rmin, load=0.0, gravity=0.0):
        super().__init__(nelx, nely, volfrac, penal, rmin)
        self.name = 'Self-weight'

        # Solution vectors
        self.f = np.zeros(self.ndof)
        self.u = np.zeros((self.ndof, 1))

        # Applied load at top
        self.load = load  # load
        self.gravity = gravity / (nelx * nely)  # Gravity force

    def g(self, x):
        g_j = np.zeros(self.m + 1)

        # Filter design variables
        xPhys = np.asarray(self.H * x[np.newaxis].T / self.Hs)[:, 0]

        # Gravity load
        self.f[:] = 0
        np.add.at(self.f, self.edofMat[:, 1::2].flatten(), np.kron(xPhys, -self.gravity * np.ones(4) / 4))

        # Applied load
        self.f[self.dout] -= self.load

        K = self.assemble_K(xPhys, interpolation='simplin')
        self.u[self.free, :] = self.linear_solve(K, self.f[self.free])

        # Objective and volume constraint
        self.ce[:] = (np.dot(self.u[self.edofMat].reshape(self.nelx * self.nely, 8), self.KE) *
                      self.u[self.edofMat].reshape(self.nelx * self.nely, 8)).sum(1)

        g_j[0] = np.asscalar(self.f @ self.u)
        g_j[1] = 1 - sum(xPhys[:]) / (self.volfrac * self.n)
        return g_j

    def dg(self, x):
        dg_j = np.zeros((self.m + 1, self.n))

        # Filter design variables
        xPhys = np.asarray(self.H * x[np.newaxis].T / self.Hs)[:, 0]

        dg_j[0, :] -= (1 - self.Eps) * (0.1 + 0.9 * self.penal * xPhys ** (self.penal - 1)) * self.ce

        # np.add.at(self.f, self.edofMat[:, 1::2].flatten(), np.kron(xPhys, -self.gravity*np.ones(4)/4))
        dg_j[0, :] -= self.u[self.edofMat[:, 1], 0] * self.gravity / 2
        dg_j[0, :] -= self.u[self.edofMat[:, 3], 0] * self.gravity / 2
        dg_j[0, :] -= self.u[self.edofMat[:, 5], 0] * self.gravity / 2
        dg_j[0, :] -= self.u[self.edofMat[:, 7], 0] * self.gravity / 2

        dg_j[1, :] = -np.ones(self.nely * self.nelx) / (self.volfrac * self.n)

        # Sensitivity filtering
        dg_j[0, :] = np.asarray(self.H * (dg_j[0, :][np.newaxis].T / self.Hs))[:, 0]
        dg_j[1, :] = np.asarray(self.H * (dg_j[1, :][np.newaxis].T / self.Hs))[:, 0]

        return dg_j


if __name__ == "__main__":
    prob = Selfweight(10, 5, 0.3, 3, 2, 1)
    x = np.random.rand(prob.n) * 1.0
    g0 = prob.g(x)
    dg_an = prob.dg(x)

    dx = 1e-4
    dg_fd = np.zeros_like(dg_an)
    for i in range(prob.n):
        x0 = x[i]
        x[i] += dx
        gp = prob.g(x)
        x[i] = x0
        dg_fd[:, i] = (gp - g0) / dx
        print(f"an: {dg_an[:, i]}, fd: {dg_fd[:, i]}, diff = {dg_an[:, i] / dg_fd[:, i] - 1.0}")
