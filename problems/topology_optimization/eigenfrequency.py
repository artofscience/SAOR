import numpy as np
from scipy.sparse.linalg import eigsh, splu, LinearOperator

import util.to_utils as utils
from sao.problems.problem import Problem


class EigenfrequencyClampedBeam(Problem):

    def __init__(self, nelx, nely, volfrac=0.2, rmin=2, n_eigenvalues=20, rho=1.0e-2):
        super().__init__()
        self.name = 'EigenfrequencyMBB'
        self.Eps = 1e-10
        self.mesh = utils.Mesh(nelx, nely)
        self.factor = None
        self.m = 1
        self.rho = rho
        self.n = self.mesh.n

        self.penal = 3
        self.volfrac = volfrac
        self.x0 = self.volfrac * np.ones(self.mesh.n, dtype=float)

        self.dc = np.zeros((self.mesh.nely, self.mesh.nelx), dtype=float)
        self.ce = np.ones(self.mesh.n, dtype=float)

        self.KE = utils.element_matrix_stiffness()

        self.filter = utils.Filter(self.mesh, rmin)

        self.dofs = np.arange(self.mesh.ndof)
        left = np.union1d(self.dofs[0:self.mesh.ndofy:2], self.dofs[1:self.mesh.ndofy:2])
        right = np.union1d(self.dofs[self.mesh.ndof - self.mesh.ndofy:self.mesh.ndof:2],
                           self.dofs[self.mesh.ndof - self.mesh.ndofy + 1:self.mesh.ndof:2])
        self.fixed = np.union1d(left, right)
        self.free = np.setdiff1d(self.dofs, self.fixed)
        self.n_eig = n_eigenvalues
        self.u = np.zeros((self.mesh.ndof, self.n_eig), dtype=float)

    def g(self, x):
        g = np.zeros(self.m + 1)

        xPhys = self.filter.forward(x)

        E = self.Eps + (0.1 * xPhys.flatten() + 0.9 * (xPhys.flatten() ** self.penal)) * (1 - self.Eps)
        K = utils.assemble_K(E, self.mesh, self.fixed)

        ro = self.Eps + (1 - self.Eps) * xPhys.flatten()
        M = utils.assemble_M(ro, self.mesh, self.free, rho=self.rho, lx=1 / self.mesh.nelx, ly=self.mesh.nely)

        Kinv = splu(K)
        KinvOp = LinearOperator(K.shape, Kinv.solve)
        self.eigvals, self.u[self.free, :] = eigsh(K, M=M, k=self.n_eig, OPinv=KinvOp, sigma=0.0)

        g[0] = np.sum(1 / self.eigvals)
        g[1] = np.sum(xPhys[:]) / (self.volfrac * self.mesh.n) - 1
        return g

    def dg(self, x):
        dg = np.zeros((2, self.mesh.n), dtype=float)
        xPhys = self.filter.forward(x)

        dg_dlam = -1 / self.eigvals ** 2

        ue = self.u[self.mesh.edofMat, :]
        dg_dsK = np.einsum("E,ijE,jk,ikE->i", dg_dlam, ue, self.KE, ue)
        dg_dsM = np.einsum("E,ijE,ijE->i",
                           -self.eigvals * dg_dlam * self.rho * (1 / self.mesh.nelx) * (1 / self.mesh.nely) / 4, ue, ue)

        dg[0, :] += dg_dsK * (1 - self.Eps) * (0.1 + 0.9 * self.penal * xPhys ** (self.penal - 1))
        dg[0, :] += dg_dsM * (1 - self.Eps)
        dg[1, :] = np.ones(self.mesh.n) / (self.volfrac * self.mesh.n)

        # Sensitivity filtering
        dg[0, :] = self.filter.backward(dg[0, :])
        dg[1, :] = self.filter.backward(dg[1, :])

        return dg


if __name__ == "__main__":
    from problems.util.fd import finite_difference

    problem = EigenfrequencyClampedBeam(4, 4)
    finite_difference(problem, problem.x0, dx=1e-7)
