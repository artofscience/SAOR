import numpy as np
from scipy.sparse import coo_matrix

import util.to_utils as utils
from sao.problems.problem import Problem


class ComplianceMBB(Problem):

    def __repr__(self):
        return f'{self.__class__.__name__}( n: {self.mesh.nelx}x{self.mesh.nely}, v: {self.vf}, r: {self.fradius} )'

    def __init__(self, nx, ny, vf=0.2, fradius=2):
        super().__init__()
        self.eps = 1e-10
        self.mesh = utils.Mesh(nx, ny)
        self.factor = None
        self.m = 1
        self.n = self.mesh.n
        self.fradius = fradius

        self.penal = 3
        self.vf = vf
        self.x0 = self.vf * np.ones(self.mesh.n, dtype=float)

        self.dc = np.zeros((self.mesh.nely, self.mesh.nelx), dtype=float)
        self.ce = np.ones(self.mesh.n, dtype=float)

        self.ke = utils.element_matrix_stiffness()

        self.filter = utils.Filter(self.mesh, fradius)

        self.dofs = np.arange(self.mesh.ndof)
        self.fixed = np.union1d(self.dofs[0:2 * (self.mesh.nely + 1):2],
                                np.array([self.mesh.ndof - 1]))
        self.free = np.setdiff1d(self.dofs, self.fixed)
        self.f = np.zeros(self.mesh.ndof, dtype=float)
        self.u = np.zeros((self.mesh.ndof, 1), dtype=float)

        # Applied load at top
        self.dout = 1
        self.f[self.dout] = -1

    def g(self, x):
        g = np.zeros(self.m + 1)

        xphys = self.filter.forward(x)

        ym = self.eps + (xphys.flatten() ** self.penal) * (1 - self.eps)
        stiffness_matrix = utils.assemble_K(ym, self.mesh, self.fixed)

        self.u[self.free, :] = utils.linear_solve(stiffness_matrix, self.f[self.free])

        # Objective and volume constraint
        self.ce[:] = (np.dot(self.u[self.mesh.edofMat].reshape(self.mesh.n, 8), self.ke) *
                      self.u[self.mesh.edofMat].reshape(self.mesh.n, 8)).sum(1)

        g[0] = np.dot(self.f, self.u)
        g[1] = np.sum(xphys[:]) / (self.vf * self.mesh.n) - 1
        return g

    def dg(self, x):
        dg = np.zeros((2, self.mesh.n), dtype=float)
        xphys = self.filter.forward(x)
        dg[0, :] -= (1 - self.eps) * (self.penal * xphys ** (self.penal - 1)) * self.ce
        dg[1, :] = np.ones(self.mesh.n) / (self.vf * self.mesh.n)
        dg[0, :] = self.filter.backward(dg[0, :])
        dg[1, :] = self.filter.backward(dg[1, :])

        return dg


class Flexure(Problem):
    def __init__(self, nx, ny, vf=0.5, fradius=2):
        super().__init__()
        self.eps = 1e-10
        self.mesh = utils.Mesh(nx, ny)
        self.factor = None
        self.m = 1
        self.fradius = fradius
        self.n = self.mesh.n

        self.penal = 2
        self.vf = vf
        self.x0 = self.vf * np.ones(self.mesh.n, dtype=float)

        self.ce = np.ones((self.mesh.n, 2), dtype=float)

        self.ke = utils.element_matrix_stiffness()

        self.filter = utils.Filter(self.mesh, fradius)

        self.dofs = np.arange(self.mesh.ndof)

        topx = self.dofs[0::self.mesh.ndofy]
        topy = self.dofs[1::self.mesh.ndofy]

        top = np.union1d(topx, topy)
        bottom = np.union1d(self.dofs[self.mesh.ndofy - 2::self.mesh.ndofy],
                            self.dofs[self.mesh.ndofy - 1::self.mesh.ndofy])
        self.fixed = np.union1d(top, bottom)
        self.free = np.setdiff1d(self.dofs, self.fixed)
        self.f = np.zeros((self.mesh.ndof, 2), dtype=float)
        self.u = np.zeros((self.mesh.ndof, 2), dtype=float)

        self.u[topx, 0] = 10
        self.u[topy, 1] = 10

    def g(self, x):
        g = np.zeros(self.m + 1)

        xphys = self.filter.forward(x)

        ym = self.eps + (xphys ** self.penal) * (1 - self.eps)
        sk = ((self.ke.flatten()[np.newaxis]).T * ym).flatten(order='F')
        stiffness_matrix = coo_matrix((sk, (self.mesh.iK, self.mesh.jK)),
                                      shape=(self.mesh.ndof, self.mesh.ndof)).tocsc()

        self.u[self.free, :] = utils.linear_solve(stiffness_matrix[self.free, :][:, self.free],
                                                  -stiffness_matrix[self.free, :][:, self.fixed] * self.u[self.fixed,
                                                                                                   :])

        # Objective and volume constraint
        for i in [0, 1]:
            u = self.u[:, i]
            self.ce[:, i] = (np.dot(u[self.mesh.edofMat].reshape(self.mesh.n, 8), self.ke) *
                             u[self.mesh.edofMat].reshape(self.mesh.n, 8)).sum(1)

        g[0] = -np.sum(np.dot(self.u[:, 0], stiffness_matrix * self.u[:, 0]))
        g[1] = np.sum(np.dot(self.u[:, 1], stiffness_matrix * self.u[:, 1])) - 1
        return g

    def dg(self, x):
        dg = np.zeros((2, self.mesh.n), dtype=float)
        xphys = self.filter.forward(x)
        for i in [0, 1]:
            dg[i, :] = (1 - self.eps) * (self.penal * xphys ** (self.penal - 1)) * self.ce[:, i]
        dg[0, :] *= -1
        dg[0, :] = self.filter.backward(dg[0, :])
        dg[1, :] = self.filter.backward(dg[1, :])

        return dg


class SelfweightArch(Problem):

    def __init__(self, nelx, nely, load=0.0, gravity=10.0, volfrac=0.2, rmin=3, x0=0.5):
        super().__init__()
        self.name = 'self-weight'
        self.Eps = 1e-10
        self.mesh = utils.Mesh(nelx, nely)
        self.factor = None
        self.n = self.mesh.n

        self.penal = 3
        self.volfrac = volfrac
        self.x0 = x0 * np.ones(self.mesh.n, dtype=float)

        self.dc = np.zeros((self.mesh.nely, self.mesh.nelx), dtype=float)
        self.ce = np.ones(self.mesh.n, dtype=float)

        self.KE = utils.element_matrix_stiffness()

        self.filter = utils.Filter(self.mesh, rmin)

        self.dofs = np.arange(self.mesh.ndof)
        self.fixed = np.union1d(self.dofs[0:self.mesh.ndofy:2],
                                np.array([self.mesh.ndof - 2, self.mesh.ndof - 1]))
        self.free = np.setdiff1d(self.dofs, self.fixed)
        self.f = np.zeros(self.mesh.ndof, dtype=float)
        self.u = np.zeros((self.mesh.ndof, 1), dtype=float)

        # Applied load at top
        self.load = load
        self.dout = 1
        self.gravity = gravity / self.mesh.n

    def g(self, x):
        g = np.zeros(2)

        xPhys = self.filter.forward(x)

        # Gravity load
        self.f[:] = 0
        np.add.at(self.f, self.mesh.edofMat[:, 1::2].flatten(),
                  np.kron(xPhys, -self.gravity * np.ones(4) / 4))
        self.f[self.dout] -= self.load

        E = self.Eps + (0.1 * xPhys.flatten() + 0.9 * (xPhys.flatten() ** self.penal)) * (1 - self.Eps)
        K = utils.assemble_K(E, self.mesh, self.fixed)

        self.u[self.free, :] = utils.linear_solve(K, self.f[self.free])

        # Objective and volume constraint
        self.ce[:] = (np.dot(self.u[self.mesh.edofMat].reshape(self.mesh.n, 8), self.KE) *
                      self.u[self.mesh.edofMat].reshape(self.mesh.n, 8)).sum(1)

        g[0] = np.dot(self.f, self.u)
        g[1] = 1 - sum(xPhys[:]) / (self.volfrac * self.mesh.n)
        return g

    def dg(self, x):
        dg = np.zeros((2, self.mesh.n), dtype=float)
        xPhys = self.filter.forward(x)
        dg[0, :] -= (1 - self.Eps) * (0.1 + 0.9 * self.penal * xPhys ** (self.penal - 1)) * self.ce

        # np.add.at(self.f, self.edofMat[:, 1::2].flatten(), np.kron(xPhys, -self.gravity*np.ones(4)/4))
        dg[0, :] -= self.u[self.mesh.edofMat[:, 1], 0] * self.gravity / 2
        dg[0, :] -= self.u[self.mesh.edofMat[:, 3], 0] * self.gravity / 2
        dg[0, :] -= self.u[self.mesh.edofMat[:, 5], 0] * self.gravity / 2
        dg[0, :] -= self.u[self.mesh.edofMat[:, 7], 0] * self.gravity / 2

        dg[1, :] = -np.ones(self.mesh.n) / (self.volfrac * self.mesh.n)

        # Sensitivity filtering
        dg[0, :] = self.filter.backward(dg[0, :])
        dg[1, :] = self.filter.backward(dg[1, :])

        return dg


class SelfweightMBB(SelfweightArch):
    def __init__(self, nelx, nely, load=0.0, gravity=10.0, volfrac=0.2, rmin=2, x0=0.5):
        super().__init__(nelx, nely, load=load, gravity=gravity, volfrac=volfrac, rmin=rmin, x0=x0)
        self.fixed = np.union1d(self.dofs[0:self.mesh.ndofy:2],
                                np.array([self.mesh.ndof - 1]))
        self.free = np.setdiff1d(self.dofs, self.fixed)


if __name__ == "__main__":
    from problems.util.fd import finite_difference

    dx = 1e-7
    problem = ComplianceMBB(4, 4)
    finite_difference(problem, problem.x0, dx)

    problem = Flexure(4, 4)
    finite_difference(problem, problem.x0, dx)

    problem = SelfweightArch(4, 4)
    finite_difference(problem, problem.x0, dx)

    problem = SelfweightMBB(4, 4)
    finite_difference(problem, problem.x0, dx)
