#
import numpy as np
#
from problems.svanberg1987 import TwoBarTruss
from sao.convergence_criteria import IterationCount
from sao.move_limits import Bounds, MoveLimit, MoveLimitFraction, AdaptiveMoveLimit
from sao.problems.subproblem_func import Subproblem
from sao.solvers.allmmadual import allmmadual
from sao.solvers.allcondual import allcondual
#
from sao.util.records import Records
#from sao.function import Function
from examples.svan.twobartruss_pdip import MMA
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
def twobartruss_dual(obj,con1,con2,sub):
#
    #instantiate the problem instance
    problem = TwoBarTruss()
#
    #bounds and move limit stuff
    bounds = Bounds(xmin=problem.x_min, xmax=problem.x_max)
    movelimit = MoveLimitFraction(fraction=2)
#
    n = problem.n
    x = problem.x0
    f = problem.g(x)
    df = problem.dg(x)
#
    aux=[problem.x_min, problem.x_max]
#
    #instantiate subproblem
    funcs = [obj, con1, con2]
    subproblem = Subproblem(funcs, limits=[bounds, movelimit])
#
    #instantiate history
    history = Records(['f0','x1','x2'])
#
    converged = IterationCount(9)
    cnt=0
    while not converged:
        f = problem.g(x); df = problem.dg(x)
        history.popcol('f0',f[0]); history.popcol('x1',x[0]); history.popcol('x2',x[1])
        infeasibility = max(0.0, f[1], f[2])
        print("{}: {:.3f} {:.3f}".format(converged.iteration - 1, f[0], infeasibility))
        if (infeasibility < 0.001) and (f[0] < 1.001 * 1.51):
            print("The optimum solution found")
            break
#
        #update the approximations
        obj.setpoint(x, f[0], df[0], aux, 3, True)
        con1.setpoint(x, f[1], df[1], aux, 3, True)
        con2.setpoint(x, f[2], df[2],  aux, 3, True)
#
        #update the subproblem
        subproblem.build(x,f,df)
#
        if sub == 'allmmadual':
            x[:] = allmmadual(subproblem,funcs)[0]
        elif sub == 'allcondual':
            x[:] = allcondual(subproblem)[0]
        else:
            print('ERROR')
#
    print("\n")
#
    return history
#
if __name__ == "__main__":
    problem = TwoBarTruss()
    obj = MMA('Weight',problem.n)
    con1 = MMA('Stress 1',problem.n)
    con2 = MMA('Stress 2',problem.n)
    sub='allmmadual'
    twobartruss_dual(obj,con1,con2,sub)
