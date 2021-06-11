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
import matplotlib.patheffects as PathEffects                    # Use an outline when dark text is on dark background


class MBBBeam(Problem, ABC):

    u: np.array = NotImplemented
    f: np.array = NotImplemented

    def __init__(self, nelx, nely, volfrac=0.4, penal=3, rmin=3):
        super().__init__()
        self.Eps = 1e-9      # ratio of Emin/Emax
        self.nelx = nelx
        self.nely = nely
        self.volfrac = volfrac
        self.ndof = 2 * (self.nelx + 1) * (self.nely + 1)
        self.penal = penal
        self.rmin = rmin
        self.n = self.nely * self.nelx
        self.xmin = np.zeros(self.n, dtype=float)
        self.xmax = np.ones(self.n, dtype=float)
        self.x0 = volfrac * np.ones(self.n, dtype=float)            # np.random.rand(self.n)
        self.xold = self.xmin.copy()
        self.m = 1
        # self.g = 0                      # must be initialized to use the NGuyen/Paulino OC approach
        self.dc = np.zeros((self.nely, self.nelx), dtype=float)
        self.ce = np.ones((self.nely * self.nelx), dtype=float)
        self.din = 2 * (nely + 1) - 1
        self.dout = 1

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

        a = np.reshape(np.arange(0, self.n), (self.nelx, self.nely)).T
        b = a[0:self.rmin, 2 * self.rmin:]
        c = a[-self.rmin:, 2 * self.rmin:-2 * self.rmin]
        d = a[:-2 * self.rmin, -self.rmin:]
        padel = np.unique(np.concatenate((b.flatten(), c.flatten(), d.flatten())))
        self.Hs[padel] = np.max(self.Hs)

        # BC's and support
        self.dofs = np.arange(2 * (self.nelx + 1) * (self.nely + 1))

        self.fixed = np.union1d(self.dofs[0:2 * (self.nely + 1):2], np.array([self.ndof - 1]))
        self.free = np.setdiff1d(self.dofs, self.fixed)

    def g(self, x_k):
        ...

    def dg(self, x_k):
        ...

    def assemble_K(self, x, add=None, interpolation="simp"):
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

    def assemble_M(self, x, rho=1.0, lx=1.0, ly=1.0, lz=1.0):
        x_scale = self.Eps + (1 - self.Eps) * x
        m_E = lx*ly*lz*rho  # Mass of one element
        sM = np.kron(x_scale, np.ones(8)*m_E/4)
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
        # Assumes that matrix is in symmetric csc form
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
            plt.title(f'{self.__class__.__name__}: n = {self.n}, iter = {iteration}', fontsize=16)
            ax.set_ylabel('nely', fontsize=16)
            ax.set_xlabel('nelx', fontsize=16)
            im = ax.imshow(-xPhys.reshape((self.nelx, self.nely)).T, cmap='gray',
                           interpolation='none', norm=colors.Normalize(vmin=-1, vmax=0))
            fig.show()
            vis = [fig, ax, im]

            # # Plot the indices of xPhys
            # fig, ax = plt.subplots()
            # plt.title('xPhys indices', fontsize=16)
            # ax.set_ylabel('nely', fontsize=16)
            # ax.set_xlabel('nelx', fontsize=16)
            # values = np.arange(0, len(xPhys)).reshape((self.nelx, self.nely)).T
            # ax.imshow(values, cmap=plt.cm.Blues, interpolation='none')
            # ax.set_xticks(np.arange(0, self.nelx, 1))
            # ax.set_yticks(np.arange(0, self.nely, 1))
            # ax.set_xticks(np.arange(-.5, self.nelx+0.5, 1), minor=True)
            # ax.set_yticks(np.arange(-.5, self.nely+0.5, 1), minor=True)
            # ax.grid(which='minor', color='k', linestyle='-', linewidth=1)
            # for i in range(0, self.nelx, 3):
            #     for j in range(0, self.nely, 3):
            #         c = values[j, i]
            #         text = ax.text(i, j, str(c), va='center', ha='center', color='black')
            #         text.set_path_effects([PathEffects.withStroke(linewidth=1.2, foreground='w')])

            return vis
        else:
            fig = vis[0]
            ax = vis[1]
            im = vis[2]
            im.set_array(-xPhys.reshape((self.nelx, self.nely)).T)
            ax.set_title(f'{self.__class__.__name__}: n = {self.n}, iter = {iteration}', fontsize=16)
            fig.canvas.draw()
            return vis

    def visualize_field(self, x_k, max, iteration, vis):
        """Function to visualize current design"""

        if iteration == 0:
            plt.ion()
            fig, ax = plt.subplots()
            ax.set_title(f'{self.__class__.__name__}: n = {self.n}, iter = {iteration}', fontsize=16)
            ax.set_ylabel('nely', fontsize=16)
            ax.set_xlabel('nelx', fontsize=16)
            im = ax.imshow(x_k.reshape((self.nelx, self.nely)).T, cmap='jet',
                           interpolation='none', norm=colors.Normalize(vmin=0, vmax=max))
            fig.show()
            vis = [fig, ax, im]
            return vis
        else:
            fig = vis[0]
            ax = vis[1]
            im = vis[2]
            im.set_array(x_k.reshape((self.nelx, self.nely)).T)
            ax.set_title(f'{self.__class__.__name__}: n = {self.n}, iter = {iteration}', fontsize=16)
            fig.canvas.draw()
            return vis
