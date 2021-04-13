from .Top88 import Top88
import numpy as np
from scipy.sparse import coo_matrix


class Inverter(Top88):
    def __init__(self, nelx, nely, volfrac, penal, rmin, kin=0.01, kout=0.001):
        super().__init__(nelx, nely, volfrac, penal, rmin)
        self.name = 'Inverter'
        self.m = 1

        self.din = 2 * (nely + 1) - 1
        self.dout = 1

        istiff = np.array([self.din, self.dout])
        jstiff = np.array([self.din, self.dout])
        self.sstiff = np.array([kin, kout])
        self.iK = np.concatenate((self.iK, istiff))
        self.jK = np.concatenate((self.jK, jstiff))

        self.fixed = np.union1d(self.dofs[0:2*(nely+1):2],
                                self.dofs[-nely:-1])
        self.free = np.setdiff1d(self.dofs, self.fixed)

        # Solution and RHS vectors
        self.f = np.zeros((self.ndof, 2))
        self.u = np.zeros((self.ndof, 2))

        # Set load
        self.f[self.din, 0] = 1
        self.f[self.dout, 1] = -1

        self.xPhys = None

    def g(self, x_k):
        g_j = np.empty(self.m + 1)

        # Filter design variables
        self.xPhys = np.asarray(self.H * x_k[np.newaxis].T / self.Hs)[:, 0]

        # add additional springs
        K = self.assemble_K(self.xPhys)

        self.u[self.free, :] = self.linear_solve(K, self.f[self.free, :])
        u = self.u[:, 0]
        lag = self.u[:, 1]

        # Objective and volume constraint
        self.ce[:] = (np.dot(u[self.edofMat].reshape(self.nelx * self.nely, 8), self.KE) *
                      lag[self.edofMat].reshape(self.nelx * self.nely, 8)).sum(1)

        g_j[0] = u[self.dout]
        g_j[1] = sum(self.xPhys[:]) / (self.volfrac * self.n) - 1
        return g_j

    def dg(self, x_k):
        dg_j = np.empty((self.m + 1, self.n))

        # Note the minus before self.penal is removed!
        dg_j[0, :] = (self.penal * self.xPhys ** (self.penal - 1) * (1 - self.Eps)) * self.ce
        dg_j[1, :] = np.ones(self.nely * self.nelx) / (self.volfrac * self.n)

        # Sensitivity filtering
        dg_j[0, :] = np.asarray(self.H * (dg_j[0, :][np.newaxis].T / self.Hs))[:, 0]
        dg_j[1, :] = np.asarray(self.H * (dg_j[1, :][np.newaxis].T / self.Hs))[:, 0]

        return dg_j
