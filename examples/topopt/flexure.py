import numpy as np
from matplotlib import colors
from scipy.sparse import coo_matrix

from examples.topopt import utils
import matplotlib.pyplot as plt

from examples.topopt.compliance_mbb import optimize
from sao.problems import Subproblem
from sao.move_limits import AdaptiveMoveLimit, Bounds, MoveLimit
from sao.intervening_variables import Linear, MMA, MixedIntervening
from sao.approximations import Taylor1


class Flexure:
    def __init__(self, nx, ny, vf=0.5, fradius=2):
        self.eps = 1e-10
        self.mesh = utils.Mesh(nx, ny)
        self.factor = None
        self.m = 1
        self.fradius = fradius

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
        bottom = np.union1d(self.dofs[self.mesh.ndofy-2::self.mesh.ndofy], self.dofs[self.mesh.ndofy-1::self.mesh.ndofy])
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
        stiffness_matrix = coo_matrix((sk, (self.mesh.iK, self.mesh.jK)), shape=(self.mesh.ndof, self.mesh.ndof)).tocsc()

        self.u[self.free, :] = utils.linear_solve(stiffness_matrix[self.free, :][:, self.free], -stiffness_matrix[self.free, :][:, self.fixed] * self.u[self.fixed, :])

        # Objective and volume constraint
        for i in [0, 1]:
            u = self.u[:,i]
            self.ce[:,i] = (np.dot(u[self.mesh.edofMat].reshape(self.mesh.n, 8), self.ke) *
                      u[self.mesh.edofMat].reshape(self.mesh.n, 8)).sum(1)

        g[0] = -np.sum(np.dot(self.u[:,0],stiffness_matrix * self.u[:,0]))
        g[1] = np.sum(np.dot(self.u[:,1],stiffness_matrix * self.u[:,1])) - 1
        return g

    def dg(self, x):
        dg = np.zeros((2, self.mesh.n), dtype=float)
        xphys = self.filter.forward(x)
        for i in [0,1]:
            dg[i, :] = (1 - self.eps) * (self.penal * xphys ** (self.penal - 1)) * self.ce[:,i]
        dg[0, :] *= -1
        dg[0, :] = self.filter.backward(dg[0, :])
        dg[1, :] = self.filter.backward(dg[1, :])

        return dg


if __name__ == '__main__':
    itercount = 30
    x0 = 0.5
    nelx = 100
    nely = 100

    ## SETUP SUBPROBLEMS

    mma = Subproblem(Taylor1(MMA()), limits=[Bounds(0, 1)])
    mma.set_name("MMA_asyinit_0.2")

    mma_ml = Subproblem(Taylor1(MMA()), limits=[Bounds(0, 1), MoveLimit(0.3)])
    mma_ml.set_name("MMA_asyinit_0.2_ML_0.3")

    lin_aml = Subproblem(Taylor1(Linear()), limits=[Bounds(0, 1), AdaptiveMoveLimit(0.3)])
    lin_aml.set_name("LIN_AML_0.3")

    mix_int = MixedIntervening(nelx * nely, 2, default=Linear())
    mix_int.set_intervening(MMA(), resp=0)
    mix_mma_lin = Subproblem(Taylor1(mix_int), limits=[Bounds(0, 1)])
    mix_mma_lin.set_name("MIX_MMA_asyinit0.2_LIN")

    sub_problems = [mma, mma_ml, lin_aml, mix_mma_lin]

    figdes, axsdes = plt.subplots(len(sub_problems), sharex=True)

    fig, axs = plt.subplots(5, 2, sharex=True)
    x = range(0, itercount - 1)

    for i, sub_problem in enumerate(sub_problems):
        problem = Flexure(nelx, nely, vf=x0)
        od1, od2, xphys = optimize(problem, sub_problem, problem.x0, itercount)
        axsdes[i].imshow(-xphys.reshape((problem.mesh.nelx, problem.mesh.nely)).T, cmap='gray',
                         interpolation='none', norm=colors.Normalize(vmin=-1, vmax=0))
        axsdes[i].axis('off')
        for j, dat in enumerate(od1.transpose()):
            axs[j, 0].plot(x, dat)
        for j, dat in enumerate(od2.transpose()):
            axs[j, 1].plot(x, dat)

    axs[0, 0].legend([x.name for x in sub_problems])
    axs[0, 0].set_ylabel(r'$g_0\left[\mathbf{x}^{(k)}\right]$')

    axs[1, 0].set_ylabel(r'$g_1\left[\mathbf{x}^{(k)}\right]$')
    axs[1, 0].set_yscale('symlog', linthresh=1e-4)

    axs[2, 0].set_ylabel(r'$\left|g_0\left[\mathbf{x}^{(k)}\right] - g_0\left[\mathbf{x}^{(k-1)}\right]\right|$')
    axs[2, 0].set_yscale('log')

    axs[3, 0].set_ylabel(r'$\frac{1}{n}\sum_{i}^n \left| x_i^{(k)} - x_i^{(k-1)}\right|$')
    axs[3, 0].set_yscale('log')

    axs[4, 0].set_ylabel(r'$\max(\left| \mathbf{x}^{(k)} - \mathbf{x}^{(k-1)}\right|)$')

    axs[0, 1].set_ylabel(
        r'$\frac{g_0\left[\mathbf{x}^{(k)}\right] - g_0\left[\mathbf{x}^{(k-1)}\right]}{\frac{\partial g_0^{(k-1)}}{\partial\mathbf{x}}\cdot \Delta \mathbf{x}^{(k-1)}}$')
    axs[1, 1].set_ylabel(
        r'$\frac{g_0\left[\mathbf{x}^{(k)}\right] - g_0\left[\mathbf{x}^{(k-1)}\right]}{a}$')

    axs[2, 1].set_ylabel(r'$\frac{4}{n} \sum_{i}^n \tilde{x}_i \left(1-\tilde{x}_i\right)$')

    axs[3, 1].set_ylabel('N')

    axs[4, 1].set_ylabel(
        r'$\frac{\Delta \mathbf{x}^{(k)} \cdot \Delta \mathbf{x}^{(k-1)}}{\left\|\Delta \mathbf{x}^{(k)} \right\| \cdot \left\|\Delta \mathbf{x}^{(k-1)}\right\|}$')

    plt.show(block=True)