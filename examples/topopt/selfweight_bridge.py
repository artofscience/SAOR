import numpy as np
import sao
from examples.topopt import utils
import matplotlib.pyplot as plt
from sao.problems import Subproblem
from sao.move_limits import AdaptiveMoveLimit, MoveLimit, Bounds
from sao.intervening_variables.mma import MMA, MMAfixedasy
from sao.intervening_variables import Linear
from sao.approximations import Taylor1, SphericalTaylor2, NonSphericalTaylor2


class Bridge:

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

        # filter
        self.filter = utils.Filter(self.mesh, rmin)
        # self.filter.set_padding([self.mesh.elgrid[0:rmin, 2 * rmin:].flatten(),
        #                          self.mesh.elgrid[-rmin:, 2 * rmin:-2 * rmin].flatten(),
        #                          self.mesh.elgrid[:-2 * rmin, -rmin:].flatten()])

        self.dofs = np.arange(self.mesh.ndof)
        self.fixed = np.union1d(self.dofs[0:2 * (self.mesh.nely + 1):2],
                                np.array([self.mesh.ndof - 1]))
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

        g[0] = np.asscalar(self.f @ self.u)
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


def optimize(problem, x, itercount, type):
    sub_problem = None

    if type == 0:
        sub_problem = Subproblem(Taylor1(MMA()), limits=[Bounds(0, 1)])
    elif type == 1:
        sub_problem = Subproblem(Taylor1(MMAfixedasy()), limits=[Bounds(0, 1), AdaptiveMoveLimit(0.2)])
    elif type == 2:
        sub_problem = Subproblem(Taylor1(Linear()), limits=[Bounds(0, 1), AdaptiveMoveLimit(0.2)])
    elif type == 3:
        # sub_problem = Subproblem(limits=[Bounds(0,1), AdaptiveMoveLimit(0.2)])
        sub_problem = Subproblem(NonSphericalTaylor2(intervening=Linear()), limits=[Bounds(0, 1)])
    elif type == 4:
        sub_problem = Subproblem(SphericalTaylor2(intervening=Linear), limits=[Bounds(0, 1)])

    converged = sao.convergence_criteria.IterationCount(itercount)
    counter = 0

    # setup for plotting design
    plotter = utils.PlotDesign(problem, x)
    g = []
    M = []
    v = []
    objchange = []
    f = [0, 0]
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
        xPhys = problem.filter.forward(x)
        M.append(np.sum(4 * xPhys.flatten() * (1 - xPhys.flatten()) / problem.mesh.n))

        v.append(np.sum(xPhys.flatten()) / problem.mesh.n)
        print(counter, ":  ", f[0])
        sub_problem.build(x, f, df)

        # positive = df[0,:] >= 0
        # x[positive] = sub_problem.x_min[positive]
        # x[~positive] = sub_problem.x_max[~positive]
        # else:
        x[:] = sao.solvers.primal_dual_interior_point.pdip(sub_problem)

    fout = problem.g(x)[0]  # Calculate the performance of the final design
    print("Final design : ", fout, x, "\n")

    return g, M, objchange, v


if __name__ == '__main__':
    itercount = 10
    x0 = 0.25
    nelx = 100
    nely = 50
    no = 5

    f = np.zeros((itercount - 1, no), dtype=float)
    m = np.zeros((itercount - 1, no), dtype=float)
    df = np.zeros((itercount - 1, no), dtype=float)
    v = np.zeros((itercount - 1, no), dtype=float)
    for i in range(0, no):
        problem = Bridge(nelx, nely, load=0, gravity=1, volfrac=x0)
        f[:, i], m[:, i], df[:, i], v[:, i] = optimize(problem, problem.x0, itercount, i)

    fig, axs = plt.subplots(4, 1)
    x = range(1, itercount)
    for i in range(0, no):
        axs[0].plot(x, f[:, i])
        axs[0].set_title('Objective [0:1]')
        axs[0].legend(['MMA', 'MMA (fixed asy) + AML', 'Linear + AML', 'Spherical', 'NonSpherical'])

        axs[1].plot(x, m[:, i])
        axs[1].set_title('Measure of non-discreteness [0:1]')

        axs[2].plot(x, df[:, i])
        axs[2].set_title('Objective change [-inf:inf]')
        axs[2].set_yscale('log')

        axs[3].plot(x, v[:, i])
        axs[3].set_title('Volume fraction [0:1]')

    plt.show(block=True)
