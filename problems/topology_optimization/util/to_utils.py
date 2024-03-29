import cvxopt
import cvxopt.cholmod
import matplotlib.pyplot as plt
import numpy as np
from cvxopt import cholmod
from matplotlib import colors
from scipy.sparse import coo_matrix, diags


class PlotDesign:
    def __init__(self, problem, x):
        self.problem = problem
        xPhys = np.asarray(problem.filter.H * x[np.newaxis].T / problem.filter.Hs)[:, 0]
        plt.ion()  # acces interactive mode
        fig, ax = plt.subplots()  # obtain figure and axis objects
        # plt.title(f'{problem.__class__.__name__}: n = {problem.mesh.n}, iter = {0}', fontsize=16)
        plt.title(problem)

        ax.set_ylabel('ny', fontsize=16)
        ax.set_xlabel('nx', fontsize=16)
        im = ax.imshow(-xPhys.reshape((problem.mesh.nelx, problem.mesh.nely)).T, cmap='gray',
                       interpolation='none', norm=colors.Normalize(vmin=-1, vmax=0))
        fig.show()
        fig.canvas.flush_events()

        self.fig = fig
        self.im = im
        self.ax = ax

    def plot(self, x, counter):
        xPhys = np.asarray(self.problem.filter.H * x[np.newaxis].T / self.problem.filter.Hs)[:, 0]
        self.im.set_array(-xPhys.reshape((self.problem.mesh.nelx, self.problem.mesh.nely)).T)
        self.ax.set_title(f'{self.problem.__class__.__name__}: n = {self.problem.mesh.n}, iter = {counter}',
                          fontsize=16)
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()


class Mesh:
    def __init__(self, nelx, nely):
        self.nelx = nelx
        self.nely = nely
        self.n = nelx * nely
        self.nnod = (nelx + 1) * (nely + 1)
        self.ndof = 2 * self.nnod
        self.ndofx = 2 * (nelx + 1)
        self.ndofy = 2 * (nely + 1)

        self.edofMat = np.zeros((self.nelx * self.nely, 8), dtype=int)
        for elx in range(self.nelx):
            for ely in range(self.nely):
                el = ely + elx * self.nely
                n1 = (self.nely + 1) * elx + ely
                n2 = (self.nely + 1) * (elx + 1) + ely
                self.edofMat[el, :] = np.array(
                    [2 * n1 + 2, 2 * n1 + 3, 2 * n2 + 2, 2 * n2 + 3, 2 * n2, 2 * n2 + 1, 2 * n1, 2 * n1 + 1])

        # Construct the index pointers for the coo format
        self.iK = np.kron(self.edofMat, np.ones((8, 1), dtype=int)).flatten()
        self.jK = np.kron(self.edofMat, np.ones((1, 8), dtype=int)).flatten()

        self.elgrid = np.reshape(np.arange(0, self.n), (nelx, nely)).T


class Filter:
    def __init__(self, mesh, rmin):
        self.mesh = mesh
        self.rmin = rmin
        # Filter: Build (and assemble) the index + data vectors for the coo matrix format
        nfilter = int(mesh.nelx * mesh.nely * ((2 * (np.ceil(rmin) - 1) + 1) ** 2))
        iH = np.zeros(nfilter)
        jH = np.zeros(nfilter)
        sH = np.zeros(nfilter)
        cc = 0
        for i in range(mesh.nelx):
            for j in range(mesh.nely):
                row = i * mesh.nely + j
                kk1 = int(np.maximum(i - (np.ceil(rmin) - 1), 0))
                kk2 = int(np.minimum(i + np.ceil(rmin), mesh.nelx))
                ll1 = int(np.maximum(j - (np.ceil(rmin) - 1), 0))
                ll2 = int(np.minimum(j + np.ceil(rmin), mesh.nely))
                for k in range(kk1, kk2):
                    for l in range(ll1, ll2):
                        col = k * mesh.nely + l
                        fac = rmin - np.sqrt(((i - k) * (i - k) + (j - l) * (j - l)))
                        iH[cc] = row
                        jH[cc] = col
                        sH[cc] = np.maximum(0.0, fac)
                        cc = cc + 1

            # Finalize assembly and convert to csc format
            self.H = coo_matrix((sH, (iH, jH)), shape=(mesh.n, mesh.n)).tocsc()
            self.Hs = self.H.sum(1)

    def forward(self, x):
        return np.asarray(self.H * x[np.newaxis].T / self.Hs)[:, 0]

    def backward(self, x):
        return np.asarray(self.H * (x[np.newaxis].T / self.Hs))[:, 0]

    def set_padding(self, x):
        padel = np.unique(np.concatenate((x)))
        self.Hs[padel] = np.max(self.Hs)


def linear_solve(K, f):
    Kcoo = K.tocoo()
    K = cvxopt.spmatrix(Kcoo.data, Kcoo.row.astype(int), Kcoo.col.astype(int))
    B = cvxopt.matrix(f)
    cvxopt.cholmod.linsolve(K, B)
    return np.array(B)


class CholmodLinearSolver:
    def __init__(self):
        self.inv = None

    def linear_solve(self, K, f):
        if self.inv is None:
            self.inv = cholmod.analyze(K, ordering_method='best')
            self.inv.cholesky_inplace(K)
        return self.inv(f)


def assemble_K(x, mesh, fixed):
    KE = element_matrix_stiffness()
    sK = ((KE.flatten()[np.newaxis]).T * x).flatten(order='F')
    K = coo_matrix((sK, (mesh.iK, mesh.jK)), shape=(mesh.ndof, mesh.ndof)).tocsc()
    K = deleterowcol(K, fixed, fixed)
    return K


def assemble_M(x, mesh, free, rho=1.0, lx=1.0, ly=1.0, lz=1.0):
    m_E = lx * ly * lz * rho  # Mass of one element
    sM = np.kron(x, np.ones(8) * m_E / 4)
    xdiag = np.zeros(mesh.ndof)
    np.add.at(xdiag, mesh.edofMat.flatten(), sM)  # Assemble the diagonal
    return diags(xdiag[free])


def element_matrix_stiffness():
    E = 1
    nu = 0.3
    k = np.array([1 / 2 - nu / 6, 1 / 8 + nu / 8, -1 / 4 - nu / 12, -1 / 8 + 3 * nu / 8, -1 / 4 + nu / 12,
                  -1 / 8 - nu / 8, nu / 6, 1 / 8 - 3 * nu / 8])
    KE = E / (1 - nu ** 2) * np.array([[k[0], k[1], k[2], k[3], k[4], k[5], k[6], k[7]],
                                       [k[1], k[0], k[7], k[6], k[5], k[4], k[3], k[2]],
                                       [k[2], k[7], k[0], k[5], k[6], k[3], k[4], k[1]],
                                       [k[3], k[6], k[5], k[0], k[7], k[2], k[1], k[4]],
                                       [k[4], k[5], k[6], k[7], k[0], k[1], k[2], k[3]],
                                       [k[5], k[4], k[3], k[2], k[1], k[0], k[7], k[6]],
                                       [k[6], k[3], k[4], k[1], k[2], k[7], k[0], k[5]],
                                       [k[7], k[2], k[1], k[4], k[3], k[6], k[5], k[0]]])
    return KE


def deleterowcol(A, delrow, delcol):
    m = A.shape[0]
    keep = np.delete(np.arange(0, m), delrow)
    A = A[keep, :]
    keep = np.delete(np.arange(0, m), delcol)
    A = A[:, keep]
    return A
