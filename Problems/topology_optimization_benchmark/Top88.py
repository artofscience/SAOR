# This is top88.m : https://www.topopt.mek.dtu.dk/Apps-and-software/Topology-optimization-codes-written-in-Python

# A 165 LINE TOPOLOGY OPTIMIZATION CODE BY NIELS AAGE AND VILLADS EGEDE JOHANSEN, JANUARY 2013
from __future__ import division
import numpy as np
from scipy.sparse import coo_matrix, diags
from matplotlib import colors
import matplotlib.pyplot as plt
import cvxopt
import cvxopt.cholmod
from sao.problems.problem import Problem
from abc import ABC

## CLASS: This is the Fig. 4 of https://www.sciencedirect.com/science/article/abs/pii/S004579491500022X
class Top88(Problem, ABC):

    fixed: np.array = NotImplemented
    free: np.array = NotImplemented
    u: np.array = NotImplemented
    f: np.array = NotImplemented

    def __init__(self, nelx, nely, volfrac=0.5, penal=3.0, rmin=2.0):
        super().__init__()
        self.Eps = 1e-9 # ratio of Emin/Emax
        self.nelx = nelx
        self.nely = nely
        self.volfrac = volfrac
        self.ndof = 2 * (self.nelx + 1) * (self.nely + 1)
        self.penal = penal
        self.rmin = rmin
        self.n = self.nely * self.nelx
        self.xmin = np.zeros(self.n, dtype=float)
        self.xmax = np.ones(self.n, dtype=float)
        self.x0 = self.volfrac * np.ones(self.n, dtype=float)
        self.xold = self.xmin.copy()
        # self.g = 0                      # must be initialized to use the NGuyen/Paulino OC approach
        self.dc = np.zeros((self.nely, self.nelx), dtype=float)
        self.ce = np.ones((self.nely * self.nelx), dtype=float)

        # FE: Build the index vectors for the for coo matrix format
        self.KE = self.element_matrix_stiffness()
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

        # Finalize assembly and convert to csc format
        self.H = coo_matrix((sH, (iH, jH)), shape=(self.nelx * self.nely, self.nelx * self.nely)).tocsc()
        self.Hs = self.H.sum(1)

        # BC's and support
        self.dofs = np.arange(2 * (self.nelx + 1) * (self.nely + 1))


    def g(self, x_k):
        ...

    def dg(self, x_k):
        ...

    def assemble_K(self, x, interpolation="simp"):
        if interpolation.lower() == 'simp':
            x_scale = (self.Eps + x ** self.penal * (1 - self.Eps))
        elif interpolation.lower() == 'simplin':
            x_scale = (self.Eps + (0.1 * x + 0.9 * (x ** self.penal)) * (1 - self.Eps))
        else:
            raise RuntimeError(f"Option {interpolation} not known as a material interpolation scheme")

        # Setup and solve FE problem
        sK = ((self.KE.flatten()[np.newaxis]).T * x_scale).flatten(order='F')
        K = coo_matrix((sK, (self.iK, self.jK)), shape=(self.ndof, self.ndof)).tocsc()
        # FIXME: Here coo is converted to csc
        # Remove constrained dofs from matrix
        K = self.deleterowcol(K, self.fixed, self.fixed)  # FIXME: Here it is (was) converted back to coo?
        return K

    def assemble_M(self, x, rho=1.0):
        x_scale = self.Eps + (1 - self.Eps) * x
        sM = np.kron(x_scale, np.ones(8)*rho/4)
        xdiag = np.zeros(self.ndof)
        np.add.at(xdiag, self.edofMat.flatten(), sM)  # Assemble the diagonal
        return diags(xdiag[self.free])

    @staticmethod
    def linear_solve(K, f):
        Kcoo = K.tocoo()
        K = cvxopt.spmatrix(Kcoo.data, Kcoo.row.astype(int), Kcoo.col.astype(int))
        B = cvxopt.matrix(f)
        cvxopt.cholmod.linsolve(K, B)
        return np.array(B)

    @staticmethod
    def element_matrix_stiffness():   # FIXME Give this function a proper name please :)
        """Element stiffness matrix"""
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

    @staticmethod
    def deleterowcol(A, delrow, delcol):
        # Assumes that matrix is in symmetric csc form !
        m = A.shape[0]
        keep = np.delete(np.arange(0, m), delrow)
        A = A[keep, :]
        keep = np.delete(np.arange(0, m), delcol)
        A = A[:, keep]
        return A

    def visualize(self, x_k, iteration, vis):
        """Function to visualize current design"""
        xPhys = np.asarray(self.H * x_k[np.newaxis].T / self.Hs)[:, 0]
        if iteration == 0:
            plt.ion()
            fig, ax = plt.subplots()
            plt.title('Half MBB-beam: iter = {}'.format(iteration), fontsize=16)
            ax.set_ylabel('nely', fontsize=16)
            ax.set_xlabel('nelx', fontsize=16)
            im = ax.imshow(-xPhys.reshape((self.nelx, self.nely)).T, cmap='gray',
                           interpolation='none', norm=colors.Normalize(vmin=-1, vmax=0))
            fig.show()
            vis = [fig, ax, im]
            return vis
        else:
            im = vis[2]
            fig = vis[0]
            im.set_array(-xPhys.reshape((self.nelx, self.nely)).T)
            plt.title('Half MBB-beam: iter = {}'.format(iteration), fontsize=16)
            fig.canvas.draw()
            return vis
