# This is top88.m : https://www.topopt.mek.dtu.dk/Apps-and-software/Topology-optimization-codes-written-in-Python

# A 165 LINE TOPOLOGY OPTIMIZATION CODE BY NIELS AAGE AND VILLADS EGEDE JOHANSEN, JANUARY 2013
from __future__ import division
import numpy as np
from scipy.sparse import coo_matrix
from matplotlib import colors
import matplotlib.pyplot as plt
import cvxopt
import cvxopt.cholmod
from sao.problems.problem import Problem


## CLASS: This is the Fig. 4 of https://www.sciencedirect.com/science/article/abs/pii/S004579491500022X
class Top88(Problem):

    def __init__(self, nelx, nely, volfrac, penal, rmin, ft):
        super().__init__()
        self.Emin = 1e-9
        self.Emax = 1.0
        self.nelx = nelx
        self.nely = nely
        self.volfrac = volfrac
        self.ndof = 2 * (self.nelx + 1) * (self.nely + 1)
        self.penal = penal
        self.rmin = rmin
        self.ft = ft
        self.n = self.nely * self.nelx
        self.m = 1
        self.xmin = 1e-3 * np.ones(self.n, dtype=float)
        self.xmax = np.ones(self.n, dtype=float)
        self.x0 = self.volfrac * np.ones(self.n, dtype=float)
        self.xold = self.xmin.copy()
        # self.g = 0                      # must be initialized to use the NGuyen/Paulino OC approach
        self.dc = np.zeros((self.nely, self.nelx), dtype=float)
        self.ce = np.ones((self.nely * self.nelx), dtype=float)
        self.name = 'Top88'

        # FE: Build the index vectors for the for coo matrix format
        self.KE = self.lk()
        self.edofMat = np.zeros((self.nelx * self.nely, 8), dtype=int)
        for elx in range(self.nelx):
            for ely in range(self.nely):
                el = ely + elx * self.nely
                n1 = (self.nely + 1) * elx + ely
                n2 = (self.nely + 1) * (elx + 1) + ely
                self.edofMat[el, :] = np.array(
                    [2 * n1 + 2, 2 * n1 + 3, 2 * n2 + 2, 2 * n2 + 3, 2 * n2, 2 * n2 + 1, 2 * n1, 2 * n1 + 1])

        # Construct the index pointers for the coo format
        self.iK = np.kron(self.edofMat, np.ones((8, 1))).flatten()
        self.jK = np.kron(self.edofMat, np.ones((1, 8))).flatten()

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
        dofs = np.arange(2 * (self.nelx + 1) * (self.nely + 1))
        self.fixed = np.union1d(dofs[0:2 * (self.nely + 1):2], np.array([2 * (self.nelx + 1) * (self.nely + 1) - 1]))
        self.free = np.setdiff1d(dofs, self.fixed)

        # Solution and RHS vectors
        self.f = np.zeros((self.ndof, 1))
        self.u = np.zeros((self.ndof, 1))

        # Set load
        self.f[1, 0] = -1

    def g(self, x_k):
        g_j = np.empty(self.m + 1)

        # Filter design variables
        if self.ft == 0:
            xPhys = x_k.copy()
        elif self.ft == 1:
            xPhys = np.asarray(self.H * x_k[np.newaxis].T / self.Hs)[:, 0]

        # Setup and solve FE problem
        sK = ((self.KE.flatten()[np.newaxis]).T * (self.Emin + xPhys ** self.penal * (self.Emax - self.Emin))).flatten(order='F')
        K = coo_matrix((sK, (self.iK, self.jK)), shape=(self.ndof, self.ndof)).tocsc()

        # Remove constrained dofs from matrix
        K = self.deleterowcol(K, self.fixed, self.fixed).tocoo()

        # Solve system
        K = cvxopt.spmatrix(K.data, K.row.astype(int), K.col.astype(int))
        B = cvxopt.matrix(self.f[self.free, 0])
        cvxopt.cholmod.linsolve(K, B)
        self.u[self.free, 0] = np.array(B)[:, 0]

        # Objective and volume constraint
        self.ce[:] = (np.dot(self.u[self.edofMat].reshape(self.nelx * self.nely, 8), self.KE) *
                      self.u[self.edofMat].reshape(self.nelx * self.nely, 8)).sum(1)
        g_j[0] = ((self.Emin + xPhys ** self.penal * (self.Emax - self.Emin)) * self.ce).sum()
        g_j[1] = sum(xPhys[:]) / (self.volfrac * self.n) - 1
        return g_j

    def dg(self, x_k):
        dg_j = np.empty((self.m + 1, self.n))

        # Filter design variables
        if self.ft == 0:
            xPhys = x_k.copy()
        elif self.ft == 1:
            xPhys = np.asarray(self.H * x_k[np.newaxis].T / self.Hs)[:, 0]

        dg_j[0, :] = (-self.penal * xPhys ** (self.penal - 1) * (self.Emax - self.Emin)) * self.ce
        dg_j[1, :] = np.ones(self.nely * self.nelx) / (self.volfrac * self.n)

        # Sensitivity filtering
        if self.ft == 0:
            dg_j[0, :] = np.asarray((self.H * (x_k * dg_j[0, :]))[np.newaxis].T / self.Hs)[:, 0] / np.maximum(0.001, x_k)
        elif self.ft == 1:
            dg_j[0, :] = np.asarray(self.H * (dg_j[0, :][np.newaxis].T / self.Hs))[:, 0]
            dg_j[1, :] = np.asarray(self.H * (dg_j[1, :][np.newaxis].T / self.Hs))[:, 0]

        return dg_j

    @staticmethod
    def lk():
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
