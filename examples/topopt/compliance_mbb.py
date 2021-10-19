import numpy as np
from examples.topopt import utils
import matplotlib.pyplot as plt
from sao.problems import Subproblem
from sao.move_limits import AdaptiveMoveLimit, Bounds
from sao.intervening_variables import Linear, MMA, MixedIntervening
from sao.approximations import Taylor1
from sao.solvers.primal_dual_interior_point import pdip
from sao.convergence_criteria import IterationCount


class ComplianceMBB:

    def __repr__(self):
        return f'{self.__class__.__name__}( n: {self.mesh.nelx}x{self.mesh.nely}, v: {self.volfrac}, r: {self.rmin} )'

    def __init__(self, nelx, nely, volfrac=0.2, rmin=2):
        self.Eps = 1e-10
        self.mesh = utils.Mesh(nelx, nely)
        self.factor = None
        self.m = 1
        self.rmin = rmin

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
        self.f = np.zeros(self.mesh.ndof, dtype=float)
        self.u = np.zeros((self.mesh.ndof, 1), dtype=float)

        # Applied load at top
        self.dout = 1
        self.f[self.dout] = -1

    def g(self, x):
        g = np.zeros(self.m + 1)

        xPhys = self.filter.forward(x)

        E = self.Eps + (xPhys.flatten() ** self.penal) * (1 - self.Eps)
        K = utils.assemble_K(E, self.mesh, self.fixed)

        self.u[self.free, :] = utils.linear_solve(K, self.f[self.free])

        # Objective and volume constraint
        self.ce[:] = (np.dot(self.u[self.mesh.edofMat].reshape(self.mesh.n, 8), self.KE) *
                      self.u[self.mesh.edofMat].reshape(self.mesh.n, 8)).sum(1)

        g[0] = np.asscalar(self.f @ self.u)
        g[1] = sum(xPhys[:]) / (self.volfrac * self.mesh.n) - 1
        return g

    def dg(self, x):
        dg = np.zeros((2, self.mesh.n), dtype=float)
        xPhys = self.filter.forward(x)
        dg[0, :] -= (1 - self.Eps) * (self.penal * xPhys ** (self.penal - 1)) * self.ce
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
        sub_problem = Subproblem(Taylor1(Linear()), limits=[Bounds(0, 1), AdaptiveMoveLimit(0.3)])
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
    dfdk = []
    xold = np.zeros_like(x)
    expected = 1
    while not converged:
        counter += 1

        fold = f[0]

        ## RESPONSES
        f = problem.g(x)
        df = problem.dg(x)
        ##

        if counter == 1:
            problem.factor = f[0]

        f[0] = f[0] / problem.factor
        df[0, :] = df[0, :] / problem.factor

        objchange.append(abs(f[0] - fold))
        dfdk.append((f[0] - fold) / expected)

        plotter.plot(x, counter)
        g.append(f[0])
        v.append(f[1])
        xPhys = problem.filter.forward(x)
        M.append(np.sum(4 * xPhys.flatten() * (1 - xPhys.flatten()) / problem.mesh.n))

        print(counter, ":  ", f[0])

        ## BUILD SUBPROBLEM
        sub_problem.build(x, f, df)
        ##

        xold[:] = x[:]

        ## SOLVE SUBPROBLEM
        x[:], a = pdip(sub_problem)
        ##

        change = x - xold
        varchange.append(np.mean(np.abs(change)))
        mvarchange.append(np.max(np.abs(change)))
        it.append(a)
        expected = np.dot(df[0, :], change)

    print("Final design : ", f[0], x, "\n")

    return g, M, objchange, v, it, varchange, mvarchange, dfdk


if __name__ == '__main__':
    itercount = 4
    x0 = 0.25
    nelx = 80
    nely = 40
    no = 3

    f = np.zeros((itercount - 1, no), dtype=float)
    m = np.zeros((itercount - 1, no), dtype=float)
    df = np.zeros((itercount - 1, no), dtype=float)
    v = np.zeros((itercount - 1, no), dtype=float)
    it = np.zeros((itercount - 1, no), dtype=float)
    varchange = np.zeros((itercount - 1, no), dtype=float)
    mvarchange = np.zeros((itercount - 1, no), dtype=float)
    dfdk = np.zeros((itercount - 1, no), dtype=float)
    for i in range(0, no):
        problem = ComplianceMBB(nelx, nely, volfrac=x0)
        f[:, i], m[:, i], df[:, i], v[:, i], it[:, i], varchange[:, i], mvarchange[:, i], dfdk[:, i] = \
            optimize(problem, problem.x0, itercount, i)

    fig, axs = plt.subplots(5, 2)
    x = range(1, itercount)
    for i in range(0, no):
        axs[0, 0].plot(x, f[:, i], marker="o")
        axs[1, 0].plot(x, f[:, i] - np.min(f, axis=1), marker="o")
        axs[2, 0].plot(x, df[:, i], marker="o")
        axs[3, 0].plot(x, v[:, i], marker="o")
        axs[0, 1].plot(x, varchange[:, i], marker="o")
        axs[1, 1].plot(x, mvarchange[:, i], marker="o")
        axs[2, 1].plot(x, it[:, i], marker="o")
        axs[3, 1].plot(x, m[:, i], marker="o")
        axs[4, 0].plot(x, dfdk[:, i], marker="o")

    axs[1, 0].set_title('Objective - min(objectives)')
    axs[1, 0].set_yscale('symlog', linthresh=1e-4)

    axs[2, 0].set_title('Objective change [-inf:inf]')
    axs[2, 0].axhline(y=1e-2, color='r', linestyle='dashed')
    axs[2, 0].axhline(y=1e-3, color='r', linestyle='dashed')
    axs[2, 0].axhline(y=1e-4, color='r', linestyle='dashed')
    axs[2, 0].set_yscale('log')

    axs[3, 0].set_title('Constraint value [-inf:inf]')
    axs[3, 0].set_yscale('symlog', linthresh=1e-4)

    axs[0, 1].set_title('Mean absolute variable change [0:1]')
    axs[0, 1].axhline(y=1e-1, color='r', linestyle='dashed')
    axs[0, 1].axhline(y=1e-2, color='r', linestyle='dashed')
    axs[0, 1].axhline(y=1e-3, color='r', linestyle='dashed')
    axs[0, 1].set_yscale('log')

    axs[1, 1].set_title('Maximum absolute variable change [0:1]')

    axs[0, 0].set_title('Objective [0:1]')
    axs[0, 0].legend(['MMA', 'Linear + AML', 'Mix MMA+LIN'])
    axs[0, 0].set_yscale('log')
    axs[0, 0].set_ylabel(r'$g_0\left[\mathbf{x}^{(k)}\right]$')

    axs[2, 1].set_title('No. of inner loops [0:inf]')
    axs[3, 1].set_title('Measure of non-discreteness [0:1]')

    axs[4, 0].set_title('Expected response change: (df(k-1)/dx(k-1) * (x_k - x_k-1) )/ (f_k - f_k-1)')

    plt.show()
    figure = plt.gcf()  # get current figure
    figure.tight_layout(pad=0.01)
    figure.set_size_inches(30, 15)
    plt.savefig("compliancedata.pdf", bbox_inches='tight', dpi=100)
    plt.show(block=True)
