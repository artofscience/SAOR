import numpy as np
import sao

"""This file demonstrates different layouts for setting up
an equivalent optimization problem.
"""

"""
Let's first generate a simple problem based on the "Problem" class:
"""


class Dummy(sao.problems.Problem):
    """
    min(x)  x.x
    s.t.    sum(x) > 1
            -1 < x_i < 1,   i=1,...,n

    Solution: x_i = 1/n
    """

    def __init__(self, n):
        super().__init__()
        self.xmin = self.alpha = 1e-3 * np.ones(n)
        self.xmax = self.beta = np.ones(n)
        self.x0 = np.random.rand(n)
        np.clip(self.x0, self.xmin, self.xmax, out=self.x0)
        self.n = n
        self.m = 1
        self.f = np.zeros(n)

    def g(self, x):
        self.f = np.array([np.dot(x, x), 1 - np.sum(x)])
        return self.f

    def dg(self, x):
        return np.array([2 * x, -np.ones_like(x)])

    def ddg(self, x):
        return np.array([2 * np.ones_like(x), np.zeros_like(x)])


"""
The idea is one can setup a problem by simply calling a wrapper function.
To use the "Method of Moving Asymptotes" (Svanberg 1987), setup is simply:
"""


def mma_wrapper(n):
    problem = Dummy(n)
    x, f = sao.solvers.method_of_moving_asymptotes.mma(problem, x0=problem.x0, stop_tol=1e-2)
    print("Final design : ", f, x, "\n")


"""
Alternatively, more advanced users can write their own loop.
An example of such a loop is as follows:
"""


def mma_loop(n):
    problem = Dummy(n)
    int_var = sao.intervening_variables.MMA()
    approx = sao.approximations.Taylor1(int_var)
    sub_problem = sao.problems.Subproblem(approx)
    x = problem.x0
    converged = sao.convergence_criteria.VariableChange(x, tolerance=1e-2)

    counter = 0
    while not converged:
        counter += 1
        f = problem.g(x)
        df = problem.dg(x)
        print(counter, ":  ", f[0], x)
        sub_problem.build(x, f, df)
        x[:] = sao.solvers.primal_dual_interior_point.pdip(sub_problem)
    fout = problem.g(x)[0]  # Calculate the performance of the final design
    print("Final design : ", fout, x, "\n")


"""
For other users, it might be more convenient to use the wrapper function `optimizer.py` for the optimization loop.
An example of such a case can be found below:
"""


def main_optimizer(n):
    """
    This is an example where the optimizer wrapper function is used.
    The result should be equivalent to that of `example_polynomial_2D().`

    :return:
    """

    # Instantiate problem, solver, approximation and convergence criterion
    problem = Dummy(n)
    int_var = sao.intervening_variables.MMA()
    approximation = sao.approximations.Taylor1(int_var)
    solver = sao.solvers.primal_dual_interior_point.pdip
    x = np.array([2, 1.5])
    converged = sao.convergence_criteria.VariableChange(x, tolerance=1e-2)

    # Call the wrapper function that conducts the optimization loop
    sao.optimizer.optimize(problem, solver, approximation, converged, x0=x, plot_flag=True)       # TODO: This `converged` here is a bit non-intuitive imo


"""
Advanced users might prefer a type of approximation depend on some convergence property.
In that case one may, for example, switch between approximation during the optimziation.
Note in the following example the move limit and approximations are generated outside of the loop,
whereas the subproblem is re-constructed each iteration. This looks as follows:
"""


def adaptive_approximation(n):
    problem = Dummy(n)
    lim = sao.move_limits.MoveLimit(0.3)
    approx1 = sao.approximations.Taylor1(sao.intervening_variables.Linear())
    approx2 = sao.approximations.Taylor1(sao.intervening_variables.MMA())
    x = problem.x0
    converged = sao.convergence_criteria.VariableChange(x, tolerance=1e-2)

    counter = 0
    while not converged:
        counter += 1
        f = problem.g(x)
        df = problem.dg(x)
        print(counter, ":  ", f[0], x)
        sub_problem = sao.problems.Subproblem(approx1 if counter < 4 else approx2, lim)
        sub_problem.build(x, f, df)
        x[:] = sao.solvers.primal_dual_interior_point.pdip(sub_problem)
    fout = problem.g(x)[0]
    print("Final design : ", fout, x, "\n")


"""
The framework also allows for easy conditional acceptance. 
In the following example we demonstrate how to adapt the loop to 
only allow designs that improve the objective:
"""


def conditional_acceptance(n):
    problem = Dummy(n)
    approx = sao.approximations.Taylor1(sao.intervening_variables.Linear())
    lim = sao.move_limits.MoveLimit()
    lim.max_dx = 0.3
    sub_problem = sao.problems.Subproblem(approx, lim)
    x = problem.x0
    converged = sao.convergence_criteria.VariableChange(x, tolerance=1e-2)

    counter = 0
    while not converged:
        counter += 1
        f = problem.g(x)
        df = problem.dg(x)
        print(counter, ":  ", "g0 ", f[0], "var ", x)
        f2 = f[0] + 1
        while f2 > f[0]:
            print(lim.max_dx)
            sub_problem.build(x, f, df)
            x_temp = sao.solvers.primal_dual_interior_point.pdip(sub_problem)
            f2 = problem.g(x_temp)[0]
            lim.max_dx *= 0.5
        x[:] = x_temp
        lim.max_dx = 0.3
    print("Final design: ", x, "\n")


if __name__ == "__main__":
    mma_wrapper(4)
    mma_loop(4)
    adaptive_approximation(4)
    conditional_acceptance(4)
