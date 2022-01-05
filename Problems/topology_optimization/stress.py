
from ..topology_optimization import utils
import numpy as np
from scipy.sparse import coo_matrix

class StressCantilever:

    def __init__(self, nx, ny, vf=0.2, fradius=3, max_stress=0.1):
        self.eps = 1e-10
        self.mesh = utils.Mesh(nx, ny)
        self.factor = None
        self.m = 1
        self.fradius = fradius

        self.max_stress = max_stress

        self.penal = 3
        self.vf = vf
        self.P = 4
        self.x0 = np.ones(self.mesh.n, dtype=float)

        self.dc = np.zeros((self.mesh.nely, self.mesh.nelx), dtype=float)
        self.ce = np.ones(self.mesh.n, dtype=float)

        self.ke = utils.element_matrix_stiffness()

        self.filter = utils.Filter(self.mesh, fradius)

        self.dofs = np.arange(self.mesh.ndof)
        self.fixed = np.union1d(self.dofs[0:self.mesh.ndof:self.mesh.ndofy],
                                self.dofs[1:self.mesh.ndof:self.mesh.ndofy])
        self.free = np.setdiff1d(self.dofs, self.fixed)
        self.f = np.zeros(self.mesh.ndof, dtype=float)
        self.u = np.zeros((self.mesh.ndof, 1), dtype=float)

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

        self.lag = np.zeros((self.mesh.ndof, 1))

        # Applied load at top
        self.dout = self.dofs[self.mesh.ndofy * (self.mesh.nelx // 2 + 1) - 2]
        self.f[self.dout] = 1

    def g(self, x):
        g = np.zeros(self.m + 1)

        xphys = self.filter.forward(x)

        ym = self.eps + (1.0 * xphys.flatten() + 0.9 * (xphys.flatten() ** self.penal)) * (1 - self.eps)
        self.stiffness_matrix = utils.assemble_K(ym, self.mesh, self.fixed)

        self.u[self.free, :] = utils.linear_solve(self.stiffness_matrix, self.f[self.free])

        self.elemental_strain = self.B.dot(self.u.flatten()[self.mesh.edofMat].transpose())
        self.elemental_strain[2, :] *= 2  # voigt notation

        self.elemental_stress = self.D.dot(self.elemental_strain).transpose()

        self.stress_vm0 = (self.elemental_stress.dot(self.V) * self.elemental_stress).sum(1)
        self.stress_vm = np.sqrt(self.stress_vm0)

        # self.stress[:] = xphys * self.stress_vm.flatten()
        self.gi = (self.stress_vm / self.max_stress) - 1
        self.gi_scaled = xphys * self.gi
        self.giplus = self.gi_scaled + 1
        self.giP = self.giplus ** self.P
        self.gisum = (1 / self.mesh.n) * np.sum(self.giP)
        giPP = self.gisum ** (1 / self.P)
        g[1] = giPP - 1

        g[0] = np.sum(xphys[:]) / self.mesh.n
        return g

    def dg(self, x):
        dg = np.zeros((2, self.mesh.n), dtype=float)
        xphys = self.filter.forward(x)

        dgdgi_scaled = (1 / self.mesh.n) * self.gisum ** (1 / self.P - 1) * self.giplus ** (self.P - 1)
        dgidstress = dgdgi_scaled[:, np.newaxis] * xphys[:, np.newaxis] * \
                     (self.stress_vm0 ** (-0.5) / self.max_stress)[:, np.newaxis] * \
                     self.elemental_stress.dot(self.V)
        dgdsstrainmat = np.einsum('jk,kl->jl', dgidstress, self.D)
        dgdsstrainmat[:, 2] *= 2
        dgdue = np.einsum('ij,jl->il', dgdsstrainmat, self.B)
        y = np.zeros(self.mesh.ndof)
        for i in range(0, self.mesh.n):
            y[self.mesh.edofMat[i, :]] += dgdue[i, :]

        self.lag[self.free, :] = utils.linear_solve(self.stiffness_matrix, y[self.free])
        self.ce[:] = (np.dot(self.u[self.mesh.edofMat].reshape(self.mesh.n, 8), self.ke) *
                      self.lag[self.mesh.edofMat].reshape(self.mesh.n, 8)).sum(1)

        dg[1, :] = -(1 - self.eps) * (1.0 + 0.9 * self.penal * xphys ** (self.penal - 1)) * self.ce
        dg[1, :] += (dgdgi_scaled * self.gi)

        dg[0, :] = np.ones(self.mesh.n) / self.mesh.n
        dg[0, :] = self.filter.backward(dg[0, :])
        dg[1, :] = self.filter.backward(dg[1, :])

        return dg
