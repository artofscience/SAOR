import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse.linalg import eigsh, splu, LinearOperator

from examples.topopt import utils
from sao.problems import Subproblem
from sao.move_limits import AdaptiveMoveLimit, Bounds
from sao.intervening_variables import Linear, MMA, MixedIntervening
from sao.approximations import Taylor1
from sao.solvers.primal_dual_interior_point import pdip
from sao.convergence_criteria import IterationCount


class EigenfrequencyMBB:

    def __init__(self, nelx, nely, volfrac=0.2, rmin=2, n_eigenvalues=3, rho=1.0):
        self.name = 'EigenfrequencyMBB'
        self.Eps = 1e-10
        self.mesh = utils.Mesh(nelx, nely)
        self.factor = None
        self.m = 1
        self.rho = rho

        self.penal = 3
        self.volfrac = volfrac
        self.x0 = self.volfrac * np.ones(self.mesh.n, dtype=float)

        self.dc = np.zeros((self.mesh.nely, self.mesh.nelx), dtype=float)
        self.ce = np.ones(self.mesh.n, dtype=float)

        self.KE = utils.element_matrix_stiffness()

        self.filter = utils.Filter(self.mesh, rmin)

        self.dofs = np.arange(self.mesh.ndof)
        self.fixed = np.union1d(self.dofs[0:2 * (self.mesh.nely + 1):2],
                                np.array([self.mesh.ndof - 1]))
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


def optimize(problem, x, itercount, type):
    sub_problem = None
    if type == 0:

        sub_problem = Subproblem(Taylor1(MMA(asyinit=0.2)), limits=[Bounds(0, 1)])
    elif type == 1:
        sub_problem = Subproblem(Taylor1(Linear()), limits=[Bounds(0, 1), AdaptiveMoveLimit(0.2)])
    elif type == 2:
        mix = MixedIntervening(problem.mesh.n, problem.m + 1, default=Linear())
        mix.set_intervening(MMA(asyinit=0.2), resp=0)
        sub_problem = Subproblem(Taylor1(mix), limits=[Bounds(0, 1)])

    converged = IterationCount(itercount)
    counter = 0

    # setup for plotting design
    plotter = utils.PlotDesign(problem, x)
    g = []
    M = []
    v = []
    it = []
    objchange = []
    f = [0, 0]
    varchange = []
    mvarchange = []
    xold = np.zeros_like(x)
    while not converged:
        counter += 1

        fold = f[0]
        f = problem.g(x)
        df = problem.dg(x)

        if counter == 1:
            problem.factor = f[0]

        f[0] = f[0] / problem.factor
        df[0, :] = df[0, :] / problem.factor

        objchange.append(abs(f[0] - fold))

        plotter.plot(x, counter)
        g.append(f[0])
        v.append(f[1])
        xPhys = problem.filter.forward(x)
        M.append(np.sum(4 * xPhys.flatten() * (1 - xPhys.flatten()) / problem.mesh.n))

        print(counter, ":  ", f[0])
        sub_problem.build(x, f, df)

        xold[:] = x[:]
        x[:], a = pdip(sub_problem)
        change = np.abs(xold - x)
        varchange.append(np.mean(change))
        mvarchange.append(np.max(change))

        it.append(a)

    print("Final design : ", f[0], x, "\n")

    return g, M, objchange, v, it, varchange, mvarchange


if __name__ == '__main__':
    itercount = 80
    x0 = 0.4
    nelx = 160
    nely = 40
    no = 3

    f = np.zeros((itercount - 1, no), dtype=float)
    m = np.zeros((itercount - 1, no), dtype=float)
    df = np.zeros((itercount - 1, no), dtype=float)
    v = np.zeros((itercount - 1, no), dtype=float)
    it = np.zeros((itercount - 1, no), dtype=float)
    varchange = np.zeros((itercount - 1, no), dtype=float)
    mvarchange = np.zeros((itercount - 1, no), dtype=float)
    for i in range(0, no):
        problem = EigenfrequencyMBB(nelx, nely, volfrac=x0, rho=1000)
        f[:, i], m[:, i], df[:, i], v[:, i], it[:, i], varchange[:, i], mvarchange[:, i] = \
            optimize(problem, problem.x0, itercount, i)

    fig, axs = plt.subplots(4, 2)
    x = range(1, itercount)
    for i in range(0, no):
        axs[0, 0].plot(x, f[:, i], marker="o")
        axs[0, 0].set_title('Objective [0:1]')
        axs[0, 0].legend(['MMA', 'Linear + AML', 'Mix MMA+LIN'])
        axs[0, 0].set_yscale('log')

        axs[1, 0].plot(x, f[:, i] - np.min(f, axis=1), marker="o")
        axs[1, 0].set_title('Objective - min(objectives)')
        axs[1, 0].set_yscale('symlog', linthresh=1e-4)

        axs[2, 0].plot(x, df[:, i], marker="o")
        axs[2, 0].set_title('Objective change [-inf:inf]')
        axs[2, 0].axhline(y=1e-2, color='r', linestyle='dashed')
        axs[2, 0].axhline(y=1e-3, color='r', linestyle='dashed')
        axs[2, 0].axhline(y=1e-4, color='r', linestyle='dashed')
        axs[2, 0].set_yscale('log')

        axs[3, 0].plot(x, v[:, i], marker="o")
        axs[3, 0].set_title('Constraint value [-inf:inf]')
        axs[3, 0].set_yscale('symlog', linthresh=1e-4)

        axs[0, 1].plot(x, varchange[:, i], marker="o")
        axs[0, 1].set_title('Mean absolute variable change [0:1]')
        axs[0, 1].axhline(y=1e-1, color='r', linestyle='dashed')
        axs[0, 1].axhline(y=1e-2, color='r', linestyle='dashed')
        axs[0, 1].axhline(y=1e-3, color='r', linestyle='dashed')
        axs[0, 1].set_yscale('log')

        axs[1, 1].plot(x, mvarchange[:, i], marker="o")
        axs[1, 1].set_title('Maximum absolute variable change [0:1]')
        axs[1, 1].axhline(y=0.2, color='r', linestyle='dashed')
        axs[1, 1].axhline(y=0.1, color='r', linestyle='dashed')
        axs[1, 1].axhline(y=0.05, color='r', linestyle='dashed')

        axs[2, 1].plot(x, it[:, i], marker="o")
        axs[2, 1].set_title('No. of inner loops [0:inf]')

        axs[3, 1].plot(x, m[:, i], marker="o")
        axs[3, 1].set_title('Measure of non-discreteness [0:1]')

    plt.show()
    figure = plt.gcf()  # get current figure
    figure.tight_layout(pad=0.1)
    figure.set_size_inches(20, 15)
    plt.savefig("eigenfrequencydata.pdf", bbox_inches='tight', dpi=100)
    plt.show(block=True)
