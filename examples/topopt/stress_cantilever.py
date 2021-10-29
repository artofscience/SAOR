import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from examples.topopt import utils
from examples.topopt.compliance_mbb import optimize
from sao.problems import Subproblem
from sao.move_limits import AdaptiveMoveLimit, Bounds, MoveLimit
from sao.intervening_variables import Linear, MMA, MixedIntervening
from sao.approximations import Taylor1


class StressCantilever:

    def __init__(self, nx, ny, vf=0.2, fradius=2, max_stress=0.5):
        self.eps = 1e-10
        self.mesh = utils.Mesh(nx, ny)
        self.factor = None
        self.m = 1
        self.fradius = fradius

        self.max_stress = max_stress

        self.penal = 3
        self.vf = vf
        self.P = 2
        self.x0 = np.ones(self.mesh.n, dtype=float)

        self.dc = np.zeros((self.mesh.nely, self.mesh.nelx), dtype=float)
        self.ce = np.ones(self.mesh.n, dtype=float)

        self.ke = utils.element_matrix_stiffness()

        self.filter = utils.Filter(self.mesh, fradius)

        self.dofs = np.arange(self.mesh.ndof)
        self.fixed = np.union1d(self.dofs[0:self.mesh.ndof:self.mesh.ndofy],
                                self.dofs[1:self.mesh.ndof:self.mesh.ndofy])
        self.free = np.setdiff1d(self.dofs, self.fixed)
        self.f = np.zeros(self.mesh.ndof, dtype=float)
        self.u = np.zeros((self.mesh.ndof, 1), dtype=float)

        self.B = 0.5 * np.array(
            [[-1, 0, 1, 0, 1, 0, -1, 0],
             [0, -1, 0, -1, 0, 1, 0, 1],
             [-1, -1, -1, 1, 1, 1, 1, -1]])

        nu = 0.3
        self.D = 1 / ((1 + nu) * (1 - 2 * nu)) * np.array(
            [[1 - nu, nu, 0],
             [nu, 1 - nu, 0],
             [0, 0, (1 - 2 * nu) / 2]])

        self.V = np.array([[1, -0.5, 0],
                           [-0.5, 1, 0],
                           [0, 0, 3]])

        self.lag = np.zeros((self.mesh.ndof, 1))

        # Applied load at top
        self.dout = self.dofs[self.mesh.ndofy * (self.mesh.nelx // 2 + 1) - 2]
        self.f[self.dout] = 1

    def g(self, x):
        g = np.zeros(self.m + 1)

        xphys = self.filter.forward(x)

        ym = self.eps + (xphys.flatten() ** self.penal) * (1 - self.eps)
        self.stiffness_matrix = utils.assemble_K(ym, self.mesh, self.fixed)

        self.u[self.free, :] = utils.linear_solve(self.stiffness_matrix, self.f[self.free])

        self.elemental_strain = self.B.dot(self.u.flatten()[self.mesh.edofMat].transpose())
        self.elemental_strain[2, :] *= 2  # voigt notation

        self.elemental_stress = self.D.dot(self.elemental_strain).transpose()

        self.stress_vm0 = (self.elemental_stress.dot(self.V) * self.elemental_stress).sum(1)
        self.stress_vm = np.sqrt(self.stress_vm0)

        # self.stress[:] = xphys * self.stress_vm.flatten()
        self.gi = (self.stress_vm / self.max_stress) - 1
        self.gi_scaled = xphys * self.gi
        self.giplus = self.gi_scaled + 1
        self.giP = self.giplus ** self.P
        self.gisum = (1 / self.mesh.n) * np.sum(self.giP)
        giPP = self.gisum ** (1 / self.P)
        g[1] = giPP - 1

        g[0] = np.sum(xphys[:]) / self.mesh.n
        return g

    def dg(self, x):
        dg = np.zeros((2, self.mesh.n), dtype=float)
        xphys = self.filter.forward(x)

        dgdgi_scaled = (1 / self.mesh.n) * self.gisum ** (1 / self.P - 1) * self.giplus ** (self.P - 1)
        dgidstress = dgdgi_scaled[:, np.newaxis] * xphys[:, np.newaxis] * \
                     (self.stress_vm0 ** (-0.5) / self.max_stress)[:, np.newaxis] * \
                     self.elemental_stress.dot(self.V)
        dgdsstrainmat = np.einsum('jk,kl->jl', dgidstress, self.D)
        dgdsstrainmat[:, 2] *= 2
        dgdue = np.einsum('ij,jl->il', dgdsstrainmat, self.B)
        y = np.zeros(self.mesh.ndof)
        for i in range(0, self.mesh.n):
            y[self.mesh.edofMat[i, :]] += dgdue[i, :]

        self.lag[self.free, :] = utils.linear_solve(self.stiffness_matrix, y[self.free])
        self.ce[:] = (np.dot(self.u[self.mesh.edofMat].reshape(self.mesh.n, 8), self.ke) *
                      self.lag[self.mesh.edofMat].reshape(self.mesh.n, 8)).sum(1)

        dg[1, :] = (-self.penal * xphys ** (self.penal - 1) * (1 - self.eps)) * self.ce
        dg[1, :] += (dgdgi_scaled * self.gi)

        dg[0, :] = np.ones(self.mesh.n) / self.mesh.n
        dg[0, :] = self.filter.backward(dg[0, :])
        dg[1, :] = self.filter.backward(dg[1, :])

        return dg


if __name__ == '__main__':
    itercount = 50
    nelx = 40
    nely = 100

    ## SETUP SUBPROBLEMS

    mma = Subproblem(Taylor1(MMA(asyinit=0.2)), limits=[Bounds(0, 1)])
    mma.set_name("MMA_asyinit_0.2")

    mma_ml = Subproblem(Taylor1(MMA(asyinit=0.2)), limits=[Bounds(0, 1), MoveLimit(0.1)])
    mma_ml.set_name("MMA_asyinit_0.2_ML_0.3")

    lin_aml = Subproblem(Taylor1(Linear()), limits=[Bounds(0, 1), AdaptiveMoveLimit(0.1)])
    lin_aml.set_name("LIN_AML_0.3")

    mix_int = MixedIntervening(nelx * nely, 2, default=MMA(asyinit=0.2))
    mix_int.set_intervening(Linear(), resp=0)
    mix_mma_lin = Subproblem(Taylor1(mix_int), limits=[Bounds(0, 1)])
    mix_mma_lin.set_name("MIX_MMA_asyinit0.2_LIN")

    sub_problems = [mma, mma_ml, lin_aml, mix_mma_lin]

    figdes, axsdes = plt.subplots(len(sub_problems), sharex=True)

    fig, axs = plt.subplots(5, 2, sharex=True)
    x = range(0, itercount - 1)

    for i, sub_problem in enumerate(sub_problems):
        problem = StressCantilever(nelx, nely)
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
    plt.savefig("stressdata.pdf", bbox_inches='tight', dpi=100)
    plt.show(block=True)
