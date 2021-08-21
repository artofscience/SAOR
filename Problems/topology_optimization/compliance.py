from .mbbbeam import MBBBeam
import numpy as np


class Compliance(MBBBeam):
    def __init__(self, nelx, nely, volfrac, penal, rmin):
        super().__init__(nelx, nely, volfrac, penal, rmin)
        self.name = 'Compliance'

        # Solution and RHS vectors
        self.f = np.zeros((self.ndof, 1))
        self.u = np.zeros((self.ndof, 1))

        # Set load
        self.f[self.dout, 0] = -1

    def g(self, x):
        g_j = np.zeros(self.m + 1)

        # Filter design variables
        xPhys = np.asarray(self.H * x[np.newaxis].T / self.Hs)[:, 0]

        K = self.assemble_K(xPhys)
        self.u[self.free, :] = self.linear_solve(K, self.f[self.free, :])

        # Objective and volume constraint
        self.ce[:] = (np.dot(self.u[self.edofMat].reshape(self.nelx * self.nely, 8), self.KE) *
                      self.u[self.edofMat].reshape(self.nelx * self.nely, 8)).sum(1)

        g_j[0] = ((self.Eps + xPhys ** self.penal * (1 - self.Eps)) * self.ce).sum()
        g_j[1] = sum(xPhys[:]) / (self.volfrac * self.n) - 1
        return g_j

    def dg(self, x):
        dg_j = np.empty((self.m + 1, self.n))

        # Filter design variables
        xPhys = np.asarray(self.H * x[np.newaxis].T / self.Hs)[:, 0]

        dg_j[0, :] = (-self.penal * xPhys ** (self.penal - 1) * (1 - self.Eps)) * self.ce
        dg_j[1, :] = np.ones(self.nely * self.nelx) / (self.volfrac * self.n)

        # Sensitivity filtering
        dg_j[0, :] = np.asarray(self.H * (dg_j[0, :][np.newaxis].T / self.Hs))[:, 0]
        dg_j[1, :] = np.asarray(self.H * (dg_j[1, :][np.newaxis].T / self.Hs))[:, 0]

        return dg_j


if __name__ == "__main__":
    prob = Compliance(20, 10, 0.3, 3, 2)
    x = np.random.rand(prob.n)*1.0
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
        print(f"an: {dg_an[:, i]}, fd: {dg_fd[:, i]}, diff = {dg_an[:, i]/dg_fd[:, i] - 1.0}")


