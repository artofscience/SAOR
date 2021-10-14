import numpy as np
import sao
from examples.topopt import utils
import matplotlib.pyplot as plt


class Bridge:

    def __init__(self, nelx, nely, load=0.0, gravity=0.0, volfrac=0.2, rmin=2):
        self.name = 'self-weight'
        self.Eps = 1e-6
        self.mesh = utils.Mesh(nelx, nely)

        self.penal = 3
        self.volfrac = volfrac
        self.x0 = volfrac * np.ones(self.mesh.n, dtype=float)

        self.dc = np.zeros((self.mesh.nely, self.mesh.nelx), dtype=float)
        self.ce = np.ones(self.mesh.n, dtype=float)

        self.KE = utils.element_matrix_stiffness()

        # filter
        self.filter = utils.Filter(self.mesh, rmin)
        self.filter.set_padding([self.mesh.elgrid[0:rmin, 2 * rmin:].flatten(),
                                 self.mesh.elgrid[-rmin:, 2 * rmin:-2 * rmin].flatten(),
                                 self.mesh.elgrid[:-2 * rmin, -rmin:].flatten()])

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


def mma(problem, x):
    int_var = sao.intervening_variables.MMA()
    approx = sao.approximations.Taylor1(int_var)
    sub_problem = sao.problems.Subproblem(approx, limits=[sao.move_limits.Bounds(0, 1)])
    converged = sao.convergence_criteria.IterationCount(20)

    counter = 0

    # setup for plotting design
    plotter = utils.PlotDesign(problem, x)
    g = []
    M = []
    while not converged:
        counter += 1

        f = problem.g(x)
        g.append(f[0])
        df = problem.dg(x)
        print(counter, ":  ", f[0], x)
        sub_problem.build(x, f, df)
        x[:] = sao.solvers.primal_dual_interior_point.pdip(sub_problem)
        plotter.plot(x, counter)
        xPhys = problem.filter.forward(x)
        M.append(100 * np.sum(4 * xPhys.flatten() * (1 - xPhys.flatten()) / problem.mesh.n))

    fout = problem.g(x)[0]  # Calculate the performance of the final design
    print("Final design : ", fout, x, "\n")

    return g, M


def linear(problem, x):
    int_var = sao.intervening_variables.Linear()
    approx = sao.approximations.Taylor1(int_var)
    bounds = sao.move_limits.Bounds(0, 1)
    aml = sao.move_limits.AdaptiveMoveLimit(move_limit=0.3)
    sub_problem = sao.problems.Subproblem(approx, limits=[bounds, aml])
    converged = sao.convergence_criteria.IterationCount(20)

    counter = 0

    # setup for plotting design
    plotter = utils.PlotDesign(problem, x)
    g = []
    M = []
    while not converged:
        counter += 1

        f = problem.g(x)
        g.append(f[0])
        df = problem.dg(x)
        print(counter, ":  ", f[0], x)
        sub_problem.build(x, f, df)
        x[:] = sao.solvers.primal_dual_interior_point.pdip(sub_problem)
        plotter.plot(x, counter)
        xPhys = problem.filter.forward(x)
        M.append(100 * np.sum(4 * xPhys.flatten() * (1 - xPhys.flatten()) / problem.mesh.n))

    fout = problem.g(x)[0]  # Calculate the performance of the final design
    print("Final design : ", fout, x, "\n")

    return g, M


if __name__ == '__main__':
    problem = Bridge(200, 100, load=0, gravity=1, volfrac=0.2)

    # first solve using standard mma
    f1, m1 = linear(problem, problem.x0)

    problem = Bridge(200, 100, load=0, gravity=1, volfrac=0.2)
    f2, m2 = mma(problem, problem.x0)

    plt.close('all')
    fig, axs = plt.subplots(2, 1)
    x = range(1, 20)
    axs[0].plot(x, f1, label="line 1")
    axs[0].plot(x, f2, label="line 2")

    axs[1].plot(x, m1, label="line 1")
    axs[1].plot(x, m2, label="line 2")
    plt.show(block=True)
