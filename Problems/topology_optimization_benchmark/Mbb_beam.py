from .Top88 import Top88
import numpy as np


class MBBBeam(Top88):
    def __init__(self, nelx, nely, volfrac, penal, rmin):
        super().__init__(nelx, nely, volfrac, penal, rmin)
        self.name = 'MBB'
        self.m = 1

        self.fixed = np.union1d(self.dofs[0:2 * (self.nely + 1):2], np.array([self.ndof - 1]))
        self.free = np.setdiff1d(self.dofs, self.fixed)

        # Solution and RHS vectors
        self.f = np.zeros((self.ndof, 1))
        self.u = np.zeros((self.ndof, 1))

        # Set load
        self.f[1, 0] = -1

        self.xPhys = None

    def g(self, x_k):

        g_j = np.empty(self.m + 1)

        # Filter design variables
        self.xPhys = np.asarray(self.H * x_k[np.newaxis].T / self.Hs)[:, 0]

        K = self.assemble_K(self.xPhys)
        self.u[self.free, :] = self.linear_solve(K, self.f[self.free, :])

        # Objective and volume constraint
        self.ce[:] = (np.dot(self.u[self.edofMat].reshape(self.nelx * self.nely, 8), self.KE) *
                      self.u[self.edofMat].reshape(self.nelx * self.nely, 8)).sum(1)

        g_j[0] = ((self.Eps + self.xPhys ** self.penal * (1 - self.Eps)) * self.ce).sum()
        g_j[1] = sum(self.xPhys[:]) / (self.volfrac * self.n) - 1
        return g_j

    def dg(self, x_k):
        dg_j = np.empty((self.m + 1, self.n))

        dg_j[0, :] = (-self.penal * self.xPhys ** (self.penal - 1) * (1 - self.Eps)) * self.ce
        dg_j[1, :] = np.ones(self.nely * self.nelx) / (self.volfrac * self.n)

        # Sensitivity filtering
        dg_j[0, :] = np.asarray(self.H * (dg_j[0, :][np.newaxis].T / self.Hs))[:, 0]
        dg_j[1, :] = np.asarray(self.H * (dg_j[1, :][np.newaxis].T / self.Hs))[:, 0]

        return dg_j

