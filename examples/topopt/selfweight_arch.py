import numpy as np
from matplotlib import colors

from examples.topopt import utils
import matplotlib.pyplot as plt
from sao.problems import Subproblem
from sao.move_limits import AdaptiveMoveLimit, Bounds, MoveLimit
from sao.intervening_variables import Linear, MMA, MixedIntervening
from sao.approximations import Taylor1

from compliance_mbb import optimize


class SelfweightArch:

    def __init__(self, nelx, nely, load=0.0, gravity=0.0, volfrac=0.2, rmin=2):
        self.name = 'self-weight'
        self.Eps = 1e-10
        self.mesh = utils.Mesh(nelx, nely)
        self.factor = None

        self.penal = 3
        self.volfrac = volfrac
        self.x0 = self.volfrac * np.ones(self.mesh.n, dtype=float)

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


if __name__ == '__main__':
    itercount = 50
    x0 = 0.2
    nelx = 50
    nely = 50

    ## SETUP SUBPROBLEMS

    mma = Subproblem(Taylor1(MMA()), limits=[Bounds(0, 1)])
    mma.set_name("MMA_asyinit_0.2")

    mma_ml = Subproblem(Taylor1(MMA()), limits=[Bounds(0, 1), MoveLimit(0.2)])
    mma_ml.set_name("MMA_asyinit_0.2_ML_0.3")

    lin_aml = Subproblem(Taylor1(Linear()), limits=[Bounds(0, 1), AdaptiveMoveLimit(0.2)])
    lin_aml.set_name("LIN_AML_0.3")

    mix_int = MixedIntervening(nelx * nely, 2, default=Linear())
    mix_int.set_intervening(MMA(), resp=0)
    mix_mma_lin = Subproblem(Taylor1(mix_int), limits=[Bounds(0, 1)])
    mix_mma_lin.set_name("MIX_MMA_asyinit0.2_LIN")

    sub_problems = [lin_aml]

    figdes, axsdes = plt.subplots(len(sub_problems), sharex=True)

    fig, axs = plt.subplots(5, 2, sharex=True)
    x = range(0, itercount - 1)

    for i, sub_problem in enumerate(sub_problems):
        problem = SelfweightArch(nelx, nely, gravity=1.0)
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

    axs[1, 1].set_ylabel(r'$\frac{4}{n} \sum_{i}^n \tilde{x}_i \left(1-\tilde{x}_i\right)$')

    axs[2, 1].set_ylabel('N')

    axs[3, 1].set_ylabel(
        r'$\arccos\left(\frac{\Delta \mathbf{x}^{(k)} \cdot \Delta \mathbf{x}^{(k-1)}}{\left\|\Delta \mathbf{x}^{(k)} \right\| \cdot \left\|\Delta \mathbf{x}^{(k-1)}\right\|}\right)$')

    plt.show()
    figure = plt.gcf()  # get current figure
    figure.tight_layout(pad=0.01)
    # figure.set_size_inches(20, 20)
    plt.savefig("selfweightdata.pdf", bbox_inches='tight', dpi=100)
    plt.show(block=True)
