from Problems.topology_optimization_benchmark.mbbbeam import MBBBeam
import numpy as np
from scipy.sparse.linalg import eigsh, splu, LinearOperator


class Eigenvalue(MBBBeam):
    def __init__(self, nelx, nely, volfrac=0.6, penal=3, rmin=2, n_eigenvalues=3, rho=1e-6):
        super().__init__(nelx, nely, volfrac, penal, rmin)
        self.unitL = 0.1/nelx

        # Solution and RHS vectors
        self.u = np.zeros((self.ndof, n_eigenvalues))

        self.n_eigenvalues = n_eigenvalues
        self.rho = rho

    def g(self, x):
        # Filter design variables
        xPhys = np.asarray(self.H * x[np.newaxis].T / self.Hs)[:, 0]

        # Assemble mass and stiffness matrix
        K = self.assemble_K(xPhys, interpolation='simplin')
        M = self.assemble_M(xPhys, rho=self.rho, lx=self.unitL, ly=self.unitL)
        # Add a point mass
        # M.data[0, 0] += 5.0
        # M.data[0, 1] += 0.5

        Kinv = splu(K)
        KinvOp = LinearOperator(K.shape, Kinv.solve)

        # Solve the eigenvalue problem
        self.eigvals, self.u[self.free, :] = eigsh(K, M=M, k=self.n_eigenvalues,  OPinv=KinvOp, sigma=0.0)
        # self.eigvals, self.u[self.free, :] = eigsh(K, M=M, k=self.n_eigenvalues,  which='SM')
        # print(f"eigenvalues found: {self.eigvals}")
        # Calculate responses
        g_j = np.empty(2)
        g_j[0] = sum(1/self.eigvals)/100
        g_j[1] = sum(xPhys[:]) / (self.volfrac * self.n) - 1
        return g_j

    def dg(self, x):
        dg_j = np.zeros((2, self.n))
        xPhys = np.asarray(self.H * x[np.newaxis].T / self.Hs)[:, 0]
        dg_dlam = -1 / self.eigvals**2

        ue = self.u[self.edofMat, :]
        dg_dsK = np.einsum("E,ijE,jk,ikE->i", dg_dlam, ue, self.KE, ue)
        dg_dsM = np.einsum("E,ijE,ijE->i", -self.eigvals*dg_dlam*self.rho/4, ue, ue)

        dg_j[0, :] += dg_dsK*(1-self.Eps)*(0.1 + 0.9*self.penal*xPhys**(self.penal-1))
        dg_j[0, :] += dg_dsM*(1-self.Eps)

        # Volume
        dg_j[1, :] = np.ones(self.nely * self.nelx) / (self.volfrac * self.n)

        # Sensitivity filtering
        dg_j[0, :] = np.asarray(self.H * (dg_j[0, :][np.newaxis].T / self.Hs))[:, 0]
        dg_j[0, :] *= 0.01
        dg_j[1, :] = np.asarray(self.H * (dg_j[1, :][np.newaxis].T / self.Hs))[:, 0]

        return dg_j


if __name__ == "__main__":
    prob = Eigenvalue(100, 10, n_eigenvalues=10)
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


