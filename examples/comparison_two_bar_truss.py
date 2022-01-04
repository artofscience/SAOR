from Problems.svanberg1987 import TwoBarTruss
from sao.move_limits.move_limit import Bounds, MoveLimit, MoveLimitST, AdaptiveMoveLimit
from sao.intervening_variables.mma import MMA87A,MMA87B, MMA87C, MMA02
from sao.intervening_variables.mixed_intervening import MixedIntervening
from sao.solvers.primal_dual_interior_point import pdip, Pdipx, Pdipxyz
from sao.intervening_variables.asymptote_update_strategies import *
from sao.problems.subproblem import Subproblem
from sao.approximations import Taylor1
from sao.convergence_criteria import IterationCount

"""
This example compares different SAO schemes for solving the Svanberg 1987 Two Bar Truss problem.
For a "fair" comparison, we use the convergence criteria as used in the paper.
"""

"""
We start with the scheme as presented in the paper.
"""
def original():
    problem = TwoBarTruss()
    bounds = Bounds(xmin=problem.x_min, xmax=problem.x_max)
    movelimit = MoveLimitST(factor=2)
    intvar = MixedIntervening(problem.n, problem.m + 1, default=MMA87A(t=0.2))
    mma_var_1 = MMA87C(sdecr=0.75, sincr=0.5,
                       x_min=problem.x_min, x_max=problem.x_max)
    intvar.set_intervening(mma_var_1, var=1)
    subproblem = Subproblem(Taylor1(intvar), limits=[bounds, movelimit])
    optimizer(problem, subproblem, IterationCount(10))

"""
Now let's see what our "default" MMA does
"""
def default_mma():
    problem = TwoBarTruss()
    bounds = Bounds(xmin=problem.x_min, xmax=problem.x_max)
    approx = Taylor1(MMA02(x_min=problem.x_min, x_max=problem.x_max))
    subproblem = Subproblem(approx, limits=[bounds, MoveLimit()])
    optimizer(problem, subproblem, IterationCount(10))

"""
What about MMA with an Adaptive Move Limit strategy?
"""
def mma_aml():
    problem = TwoBarTruss()
    bounds = Bounds(xmin=problem.x_min, xmax=problem.x_max)
    movelimit = AdaptiveMoveLimit(move_limit=0.5, dx=problem.x_max - problem.x_min)
    subproblem = Subproblem(Taylor1(MMA02(x_min=problem.x_min, x_max=problem.x_max)), limits=[bounds, movelimit])
    optimizer(problem, subproblem, IterationCount(10))

"""
Let's check out LP + AML
"""
def lp_aml():
    problem = TwoBarTruss()
    bounds = Bounds(xmin=problem.x_min, xmax=problem.x_max)
    movelimit = AdaptiveMoveLimit(move_limit=0.1, dx=problem.x_max - problem.x_min)
    subproblem = Subproblem(Taylor1(), limits=[bounds, movelimit])
    optimizer(problem, subproblem, IterationCount(10))

"""
Let's check a mixed scheme.
"""
def mixed_lp_mma():
    problem = TwoBarTruss()
    bounds = Bounds(xmin=problem.x_min, xmax=problem.x_max)
    intvar = MixedIntervening(problem.n, problem.m+1)
    intvar.set_intervening(MMA87A(t=3/4), var=1)
    subproblem = Subproblem(Taylor1(intvar), limits=[bounds])
    optimizer(problem, subproblem, IterationCount(10))

    """
Let's check a mixed scheme.
"""
def mma2():
    problem = TwoBarTruss()
    bounds = Bounds(xmin=problem.x_min, xmax=problem.x_max)
    subproblem = Subproblem(Taylor1(MMA87A(t=3/4)), limits=[bounds])
    optimizer(problem, subproblem, IterationCount(10))

def optimizer(problem, subproblem, converged):
    x = problem.x0
    while not converged:
        f = problem.g(x)
        df = problem.dg(x)
        infeasibility = max(0.0, f[1], f[2])
        print("{}: {:.3f} {:.3f}".format(converged.iteration-1, f[0], infeasibility))
        if (infeasibility < 0.001) and (f[0] < 1.001*1.51):
            print("The optimum solution found")
            break
        subproblem.build(x, f, df)
        x[:] = pdip(subproblem, variables=Pdipx)[0]
    print("\n")

if __name__ == "__main__":
    original()
    default_mma()
    mma_aml()
    lp_aml()
    mixed_lp_mma()
    mma2()
