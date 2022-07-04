#
from problems.svanberg1987 import TwoBarTruss
from sao.approximations import Taylor1, SphericalTaylor2, NonSphericalTaylor2
from sao.convergence_criteria import IterationCount
from sao.intervening_variables import Exponential
from sao.intervening_variables.mixed_intervening import MixedIntervening
from sao.intervening_variables.mma import MMA87A, MMA87C, MMA02
from sao.move_limits import Bounds, MoveLimit, MoveLimitFraction, AdaptiveMoveLimit
from sao.problems.subproblem import Subproblem
from sao.solvers.primal_dual_interior_point import pdip, Pdipx
#
from sao.util.records import Records
#
"""
This example compares different SAO schemes for solving the Svanberg 1987 Two Bar Truss problem.
For a "fair" comparison, we use the convergence criteria as used in the paper.
"""
#
"""
We start with the scheme as presented in the paper.
"""
#
def two_bar_truss():
#
    #instantiate the problem instance
    problem = TwoBarTruss()
    #from move_limits; should this not automatically come from problem definition?
    bounds = Bounds(xmin=problem.x_min, xmax=problem.x_max)
    #from move_limits: 
    movelimit = MoveLimitFraction(fraction=2)
    #intervening
    intvar = MixedIntervening(problem.n, problem.m + 1, default=MMA87A(t=0.2))
    mma_var_1 = MMA87C(sdecr=0.75, sincr=0.5, x_min=problem.x_min, x_max=problem.x_max)
    intvar.set_intervening(mma_var_1, var=1)
    #subproblem
    subproblem = Subproblem(Taylor1(intvar), limits=[bounds, movelimit])
#
    history = Records(['f0','f1','f2'])
    h=[] # make a history object
#
    x = problem.x0
    converged = IterationCount(10)
    while not converged:
        f = problem.g(x)
        history.popcol('f0',f[0])
        history.popcol('f1',f[1])
        history.popcol('f2',f[2])
        h.append(f[0])
        df = problem.dg(x)
        infeasibility = max(0.0, f[1], f[2])
        print("{}: {:.3f} {:.3f}".format(converged.iteration - 1, f[0], infeasibility))
        if (infeasibility < 0.001) and (f[0] < 1.001 * 1.51):
            print("The optimum solution found")
            break
        subproblem.build(x, f, df)
        x[:] = pdip(subproblem, variables=Pdipx)[0]
    print("\n")
#
    return history
#
if __name__ == "__main__":
    two_bar_truss()
