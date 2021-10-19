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
        return f'{self.__class__.__name__}( n: {self.mesh.nelx}x{self.mesh.nely}, v: {self.vf}, r: {self.fradius} )'

    def __init__(self, nx, ny, vf=0.2, fradius=2):
        self.eps = 1e-10
        self.mesh = utils.Mesh(nx, ny)
        self.factor = None
        self.m = 1
        self.fradius = fradius

        self.penal = 3
        self.vf = vf
        self.x0 = self.vf * np.ones(self.mesh.n, dtype=float)

        self.dc = np.zeros((self.mesh.nely, self.mesh.nelx), dtype=float)
        self.ce = np.ones(self.mesh.n, dtype=float)

        self.ke = utils.element_matrix_stiffness()

        self.filter = utils.Filter(self.mesh, fradius)

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

        xphys = self.filter.forward(x)

        ym = self.eps + (xphys.flatten() ** self.penal) * (1 - self.eps)
        stiffness_matrix = utils.assemble_K(ym, self.mesh, self.fixed)

        self.u[self.free, :] = utils.linear_solve(stiffness_matrix, self.f[self.free])

        # Objective and volume constraint
        self.ce[:] = (np.dot(self.u[self.mesh.edofMat].reshape(self.mesh.n, 8), self.ke) *
                      self.u[self.mesh.edofMat].reshape(self.mesh.n, 8)).sum(1)

        g[0] = np.dot(self.f, self.u)
        g[1] = np.sum(xphys[:]) / (self.vf * self.mesh.n) - 1
        return g

    def dg(self, x):
        dg = np.zeros((2, self.mesh.n), dtype=float)
        xphys = self.filter.forward(x)
        dg[0, :] -= (1 - self.eps) * (self.penal * xphys ** (self.penal - 1)) * self.ce
        dg[1, :] = np.ones(self.mesh.n) / (self.vf * self.mesh.n)
        dg[0, :] = self.filter.backward(dg[0, :])
        dg[1, :] = self.filter.backward(dg[1, :])

        return dg


def optimize(problem, sub_problem, x, itercount):
    converged = IterationCount(itercount)
    counter = 0
    output = np.zeros((itercount - 1, 8), dtype=float)
    # setup for plotting design
    plotter = utils.PlotDesign(problem, x)

    f = [0, 0]
    xold = np.zeros_like(x)
    expected = 1
    while not converged:

        plotter.plot(x, counter)

        fold = f[0]
        ## RESPONSES
        f = problem.g(x)
        df = problem.dg(x)
        ##

        # Scaling
        if counter == 0:
            problem.factor = f[0]
        f[0] = f[0] / problem.factor
        df[0, :] = df[0, :] / problem.factor

        print(counter, ":  ", f[0])
        output[counter, 0:2] = f[:]  # objective and constraint values
        output[counter, 2] = abs(f[0] - fold)  # objective change
        output[counter, 3] = (f[0] - fold) / expected  # relative objective change

        xphys = problem.filter.forward(x)
        output[counter, 4] = np.sum(4 * xphys.flatten() * (1 - xphys.flatten()) / problem.mesh.n)

        ## BUILD SUBPROBLEM
        sub_problem.build(x, f, df)
        ##

        xold[:] = x[:]
        ## SOLVE SUBPROBLEM
        x[:], no_inner_iter = pdip(sub_problem)
        ##
        output[counter, 5] = no_inner_iter  # number of inner iterations in pdip

        change = x - xold
        expected = np.dot(df[0, :], change)

        output[counter, 6] = np.mean(np.abs(change))  # mean variable change
        output[counter, 7] = np.max(np.abs(change))  # max variable change

        counter += 1

    return output


if __name__ == '__main__':
    itercount = 10
    x0 = 0.25
    nelx = 80
    nely = 40

    ## SETUP SUBPROBLEMS

    mma = Subproblem(Taylor1(MMA(asyinit=0.2)), limits=[Bounds(0, 1)])

    lin_aml = Subproblem(Taylor1(Linear()), limits=[Bounds(0, 1), AdaptiveMoveLimit(0.3)])

    mix_int = MixedIntervening(nelx * nely, 2, default=Linear())
    mix_int.set_intervening(MMA(asyinit=0.2), resp=0)
    mix_mma_lin = Subproblem(Taylor1(mix_int), limits=[Bounds(0, 1)])

    sub_problems = [mma, lin_aml, mix_mma_lin]

    output_data = np.zeros((itercount - 1, 8, len(sub_problems)), dtype=float)

    for i, sub_problem in enumerate(sub_problems):
        problem = ComplianceMBB(nelx, nely, vf=x0)
        output_data[:, :, i] = optimize(problem, sub_problem, problem.x0, itercount)

    fig, axs = plt.subplots(5, 2)
    x = range(0, itercount - 1)
    for i, sub_problem in enumerate(sub_problems):
        axs[0, 0].plot(x, output_data[:, 0, i], marker="o")
        axs[1, 0].plot(x, output_data[:, 0, i] - np.min(output_data[:, 0, :], axis=1), marker="o")
        axs[2, 0].plot(x, output_data[:, 1, i], marker="o")
        axs[3, 0].plot(x, output_data[:, 2, i], marker="o")
        axs[4, 0].plot(x, output_data[:, 3, i], marker="o")
        axs[0, 1].plot(x, output_data[:, 4, i], marker="o")
        axs[1, 1].plot(x, output_data[:, 5, i], marker="o")
        axs[2, 1].plot(x, output_data[:, 6, i], marker="o")
        axs[3, 1].plot(x, output_data[:, 7, i], marker="o")

    axs[0, 0].set_title('Objective [0:1]')
    axs[0, 0].legend(['mma', 'lin + aml', 'mixed'])
    axs[0, 0].set_yscale('log')
    axs[0, 0].set_ylabel(r'$g_0\left[\mathbf{x}^{(k)}\right]$')

    axs[1, 0].set_title('Objective - min(objectives)')
    axs[1, 0].set_yscale('symlog', linthresh=1e-4)

    axs[2, 0].set_title('Constraint value [-inf:inf]')
    axs[2, 0].set_yscale('symlog', linthresh=1e-4)

    axs[3, 0].set_title('Objective change [-inf:inf]')
    axs[3, 0].set_yscale('log')

    axs[4, 0].set_title('Expected response change: (f_k - f_k-1)/(df(k-1)/dx(k-1) * (x_k - x_k-1) )')
    axs[0, 1].set_title('Measure of non-discreteness [0:1]')
    axs[1, 1].set_title('No. of inner loops [0:inf]')
    axs[2, 1].set_title('Mean absolute variable change [0:1]')
    axs[0, 1].set_yscale('log')

    axs[3, 1].set_title('Maximum absolute variable change [0:1]')

    plt.show()
    figure = plt.gcf()  # get current figure
    figure.tight_layout(pad=0.01)
    figure.set_size_inches(30, 15)
    plt.savefig("compliancedata.pdf", bbox_inches='tight', dpi=100)
    plt.show(block=True)
