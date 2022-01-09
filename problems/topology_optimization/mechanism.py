import numpy as np
from scipy.sparse import coo_matrix

import util.to_utils as utils
from sao.problems import Problem


class MechanismClampedBeam(Problem):
    def __init__(self, nx, ny, vf=0.2, fradius=2, kin=100, kout=100):
        super().__init__()
        self.eps = 1e-10
        self.mesh = utils.Mesh(nx, ny)
        self.factor = None
        self.m = 1
        self.fradius = fradius
        self.n = self.mesh.n

        self.penal = 3
        self.vf = vf
        self.x0 = self.vf * np.ones(self.mesh.n, dtype=float)

        self.dc = np.zeros((self.mesh.nely, self.mesh.nelx), dtype=float)
        self.ce = np.ones(self.mesh.n, dtype=float)

        self.ke = utils.element_matrix_stiffness()

        self.filter = utils.Filter(self.mesh, fradius)

        self.dofs = np.arange(self.mesh.ndof)
        left = self.dofs[0:self.mesh.ndofy:2]
        right = np.union1d(self.dofs[self.mesh.ndof - self.mesh.ndofy // 2:self.mesh.ndof:2],
                           self.dofs[self.mesh.ndof - self.mesh.ndofy // 2 + 1:self.mesh.ndof:2])
        self.fixed = np.union1d(left, right)
        self.fixed = np.union1d(left, right)

        self.free = np.setdiff1d(self.dofs, self.fixed)
        self.f = np.zeros((self.mesh.ndof, 2), dtype=float)
        self.u = np.zeros((self.mesh.ndof, 2), dtype=float)

        self.din = self.mesh.ndofy - 1
        self.dout = 1

        self.f[self.din, 0] = 1
        self.f[self.dout, 1] = -1

        istiff = np.array([self.din, self.dout])
        jstiff = np.array([self.din, self.dout])
        self.sstiff = np.array([kin, kout])
        self.mesh.iK = np.concatenate((self.mesh.iK, istiff))
        self.mesh.jK = np.concatenate((self.mesh.jK, jstiff))

    def g(self, x):
        g = np.zeros(self.m + 1)
        # y = x.reshape(self.mesh.nelx, self.mesh.nely)
        # y[0:2,0:4] = 1
        # y[0:2,-4::] = 1
        # y[self.mesh.nely-self.mesh.nely//10:self.mesh.nely-1,0:self.mesh.nelx//10] = 1
        xphys = self.filter.forward(x.flatten())

        ym = self.eps + (xphys.flatten() ** self.penal) * (1 - self.eps)
        sk = np.concatenate((((self.ke.flatten()[np.newaxis]).T * ym).flatten(order='F'), self.sstiff))
        stiffness_matrix = coo_matrix((sk, (self.mesh.iK, self.mesh.jK)),
                                      shape=(self.mesh.ndof, self.mesh.ndof)).tocsc()

        self.u[self.free, :] = utils.linear_solve(stiffness_matrix[self.free, :][:, self.free], self.f[self.free, :])
        u = self.u[:, 0]
        lag = self.u[:, 1]

        # Objective and volume constraint
        self.ce[:] = (np.dot(u[self.mesh.edofMat].reshape(self.mesh.n, 8), self.ke) *
                      lag[self.mesh.edofMat].reshape(self.mesh.n, 8)).sum(1)

        g[0] = u[self.dout]
        g[1] = np.sum(xphys[:]) / (self.vf * self.mesh.n) - 1
        return g

    def dg(self, x):
        dg = np.zeros((2, self.mesh.n), dtype=float)
        xphys = self.filter.forward(x)
        dg[0, :] = (1 - self.eps) * (self.penal * xphys ** (self.penal - 1)) * self.ce
        dg[1, :] = np.ones(self.mesh.n) / (self.vf * self.mesh.n)
        dg[0, :] = self.filter.backward(dg[0, :])
        dg[1, :] = self.filter.backward(dg[1, :])

        return dg


if __name__ == "__main__":
    from problems.util.fd import finite_difference

    problem = MechanismClampedBeam(4, 4)
    finite_difference(problem, problem.x0, dx=1e-7)
