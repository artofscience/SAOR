from .Top88 import Top88
import numpy as np


class LBracket(Top88):
    def __init__(self, nelx, nely, volfrac, penal, rmin, max_stress=300):
        super().__init__(nelx, nely, volfrac, penal, rmin)
        self.max_stress = max_stress
        self.name = 'LBracket'
        self.m = 1

        self.fixed = np.union1d(self.dofs[0:2 * (self.nely + 1):2], np.array([self.ndof - 1]))
        self.free = np.setdiff1d(self.dofs, self.fixed)

        # Solution and RHS vectors
        self.f = np.zeros((self.ndof, 1))
        self.u = np.zeros((self.ndof, 1))
        self.stress = np.zeros(self.nel)

        # Set load
        self.f[1, 0] = -1

        self.B = 0.5 * np.array(
            [[-1, 0, 1, 0, 1, 0, -1, 0],
             [0, -1, 0, -1, 0, 1, 0, 1],
             [-1, -1, -1, 1, 1, 1, 1, -1]])

        nu = 0.3
        self.D = 1 / ((1 + nu) * (1 - 2 * nu)) * np.array(
            [[1 - nu, nu, 0],
             [nu, 1 - nu, 0],
             [0, 0, (1 - 2 * nu) / 2]])

        self.V = np.array([[1, -0.5, 0],
                           [-0.5, 1, 0],
                           [0, 0, 3]])

    def g(self, x_k):

        g_j = np.empty(self.m + 1)

        # Filter design variables
        xPhys = np.asarray(self.H * x_k[np.newaxis].T / self.Hs)[:, 0]

        K = self.fea_assembly(xPhys)
        self.u[self.free, :] = self.fea_solve(K, self.f[self.free, :])

        elemental_strain = self.B.dot(self.u.flatten()[self.edofMat].transpose())
        elemental_strain[2, :] *= 2  # voigt notation

        elemental_stress = self.D.dot(elemental_strain).transpose()
        stress_vm = np.sqrt((elemental_stress.dot(self.V) * elemental_stress).sum(1))

        self.stress[:] = xPhys * stress_vm.flatten()

        P = 10
        gi = xPhys * (stress_vm / self.max_stress - 1)
        giP = (gi + 1)**P
        giPP = (1/self.n * np.sum(giP))**(1/P)
        g_j[1] = giPP - 1
        g_j[0] = sum(xPhys[:]) / self.n
        return g_j

    def dg(self, x_k):
        dg_j = np.empty((self.m + 1, self.n))

        # Filter design variable sensitivities
        # TODO unfortunately we filter twice (both in g and dg), can we circumvent this?
        xPhys = np.asarray(self.H * x_k[np.newaxis].T / self.Hs)[:, 0]

        dg_j[0, :] = (-self.penal * xPhys ** (self.penal - 1) * (1 - self.Eps)) * self.ce
        dg_j[0, :] = np.ones(self.nely * self.nelx) / (self.volfrac * self.n)

        # Sensitivity filtering
        dg_j[0, :] = np.asarray(self.H * (dg_j[0, :][np.newaxis].T / self.Hs))[:, 0]
        dg_j[1, :] = np.asarray(self.H * (dg_j[1, :][np.newaxis].T / self.Hs))[:, 0]

        return dg_j

