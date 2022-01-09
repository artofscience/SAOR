import numpy as np
from scipy.sparse import coo_matrix

from sao.problems import Problem


class ThermoMechClamped(Problem):
    def __init__(self, nelx, nely, volfrac, penal, rmin, load=0.0, gravity=0.0):
        super().__init__(nelx, nely, volfrac, penal, rmin)
        self.name = 'Self-weight'

        # Solution vectors
        self.f = np.zeros(self.ndof)
        self.u = np.zeros((self.ndof, 1))

        # Applied load at top
        self.load = load  # load
        self.gravity = gravity / (nelx * nely)  # Gravity force
        # Construct the index pointers for the coo format
        self.iK = np.kron(self.edofMat, np.ones((8, 1), dtype=int)).flatten()
        self.jK = np.kron(self.edofMat, np.ones((1, 8), dtype=int)).flatten()

        # Filter: Build (and assemble) the index + data vectors for the coo matrix format
        nfilter = int(self.nelx * self.nely * ((2 * (np.ceil(rmin) - 1) + 1) ** 2))
        iH = np.zeros(nfilter)
        jH = np.zeros(nfilter)
        sH = np.zeros(nfilter)
        cc = 0
        for i in range(self.nelx):
            for j in range(self.nely):
                row = i * self.nely + j
                kk1 = int(np.maximum(i - (np.ceil(rmin) - 1), 0))
                kk2 = int(np.minimum(i + np.ceil(rmin), self.nelx))
                ll1 = int(np.maximum(j - (np.ceil(rmin) - 1), 0))
                ll2 = int(np.minimum(j + np.ceil(rmin), self.nely))
                for k in range(kk1, kk2):
                    for l in range(ll1, ll2):
                        col = k * self.nely + l
                        fac = rmin - np.sqrt(((i - k) * (i - k) + (j - l) * (j - l)))
                        iH[cc] = row
                        jH[cc] = col
                        sH[cc] = np.maximum(0.0, fac)
                        cc = cc + 1
        self.H = coo_matrix((sH, (iH, jH)), shape=(self.nelx * self.nely, self.nelx * self.nely)).tocsc()
        self.Hs = self.H.sum(1)
        # a = np.reshape(np.arange(0, self.n), (self.nx, self.ny)).T
        # b = a[0:self.fradius, 2 * self.fradius:]
        # c = a[-self.fradius:, 2 * self.fradius:-2 * self.fradius]
        # d = a[:-2 * self.fradius, -self.fradius:]
        # padel = np.unique(np.concatenate((b.flatten(), c.flatten(), d.flatten())))
        # self.Hs[padel] = np.max(self.Hs)

        self.fixed = np.union1d(self.dofs[0:2 * (self.nely + 1):2], self.dofs[-2 * (self.nely + 1)::])
        self.free = np.setdiff1d(self.dofs, self.fixed)

    def g(self, x):
        g_j = np.zeros(self.m + 1)

        # Filter design variables
        xPhys = np.asarray(self.H * x[np.newaxis].T / self.Hs)[:, 0]

        # Gravity load
        self.f[:] = 0
        np.add.at(self.f, self.edofMat[:, [0, 1, 3, 6]].flatten(), np.kron(xPhys, -self.gravity * np.ones(4) / 4))
        np.add.at(self.f, self.edofMat[:, [2, 4, 5, 7]].flatten(), np.kron(xPhys, self.gravity * np.ones(4) / 4))

        # Applied load
        self.f[self.dout] -= self.load

        K = self.assemble_K(xPhys, interpolation='simplin')
        self.u[self.free, :] = self.linear_solve(K, self.f[self.free])

        # Objective and volume constraint
        self.ce[:] = (np.dot(self.u[self.edofMat].reshape(self.nelx * self.nely, 8), self.KE) *
                      self.u[self.edofMat].reshape(self.nelx * self.nely, 8)).sum(1)

        g_j[0] = np.asscalar(self.f @ self.u)
        g_j[1] = sum(xPhys[:]) / (self.volfrac * self.n) - 1
        return g_j

    def dg(self, x):
        dg_j = np.zeros((self.m + 1, self.n))

        # Filter design variables
        xPhys = np.asarray(self.H * x[np.newaxis].T / self.Hs)[:, 0]

        dg_j[0, :] -= (1 - self.Eps) * (0.1 + 0.9 * self.penal * xPhys ** (self.penal - 1)) * self.ce

        # np.add.at(self.f, self.edofMat[:, 1::2].flatten(), np.kron(xPhys, -self.gravity*np.ones(4)/4))
        for i in [0, 1, 3, 6]:
            dg_j[0, :] -= self.u[self.edofMat[:, i], 0] * self.gravity * 2

        for i in [2, 4, 5, 7]:
            dg_j[0, :] += self.u[self.edofMat[:, i], 0] * self.gravity * 2

        dg_j[1, :] = np.ones(self.nely * self.nelx) / (self.volfrac * self.n)

        # Sensitivity filtering
        dg_j[0, :] = np.asarray(self.H * (dg_j[0, :][np.newaxis].T / self.Hs))[:, 0]
        dg_j[1, :] = np.asarray(self.H * (dg_j[1, :][np.newaxis].T / self.Hs))[:, 0]

        return dg_j


if __name__ == "__main__":
    from problems.util.fd import finite_difference

    problem = ThermoMechClamped(4, 4)
    finite_difference(problem, problem.x0, dx=1e-7)
