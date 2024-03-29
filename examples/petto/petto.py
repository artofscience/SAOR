#
import os
import time
#
import numpy as np
#
from sao.convergence_criteria import IterationCount
from sao.move_limits import Bounds, MoveLimit, MoveLimitFraction, AdaptiveMoveLimit
from sao.problems.subproblem_func import Subproblem
from sao.solvers.primal_dual_interior_point import pdip, Pdipx
from sao.solvers.t2dual import t2dual
from sao.solvers.osqp import osqp
#
from sao.util.records import Records
from sao.function import Function
#
from examples.petto.petsctopopt import PETScTopOpt
#from problems.svanberg1987 import CantileverBeam
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
class T2R(Function):
#
    def intercurve(self, x):
#
        x_k = self.x_k
#
        y = np.zeros_like(x)
        dy = np.zeros_like(x)
        ddy = np.zeros_like(x)
        ddy = np.zeros_like(x)
        c_x = np.zeros_like(x)
#
        dg_k = self.dg_k
        for i in range(self.n):
            c_x[i] = 2e0*abs(dg_k[i])/x_k[i]
            if dg_k[i] < 0e0:
                y[i] = x[i]
                dy[i] = 1e0
                ddy[i] = 0e0
            else:
                y[i] = x[i]
                dy[i] = 1e0
                ddy[i] = 0e0
#
        return y, dy, ddy, c_x
#
class L(Function):
#
    def intercurve(self, x):
#
        x_k = self.x_k
#
        y = np.zeros_like(x)
        dy = np.zeros_like(x)
        ddy = np.zeros_like(x)
        ddy = np.zeros_like(x)
        c_x = np.zeros_like(x)
#
        dg_k = self.dg_k
        for i in range(self.n):
            y[i] = x[i]
            dy[i] = 1e0
            ddy[i] = 0e0
            c_x[i] = 0e0
#
        return y, dy, ddy, c_x
#
def petsctopopt_t2r(sub):
#
    #instantiate the problem instance
    problem = PETScTopOpt()
#
    #bounds and move limit stuff
    bounds = Bounds(xmin=problem.x_min, xmax=problem.x_max)
    movelimit = MoveLimit(move_limit=0.2,dx=problem.x_max-problem.x_min)
#
    cnt=0

    n = problem.n
    x = problem.x0
#   problem.setx(x)
#   problem.solve(cnt)
#   f = problem.g(x)
#   df = problem.dg(x)

#   cnt=cnt+1
#
    aux=[]
#
    obj = T2R('Compliance',problem.n)
    con1 = T2R('Volume',problem.n)
#
    #instantiate subproblem
    funcs = [obj, con1]
    subproblem = Subproblem(funcs, limits=[bounds, movelimit])
#
    #instantiate history
    history = Records(['f0','inf'])
#
    converged=IterationCount(100)
#
    x_old=np.zeros_like(x)
#
    while not converged:
#
        start = time.time()
        problem.setx(x)
        problem.solve(cnt)
        f = problem.g(x); df = problem.dg(x)
        end = time.time()
        print('physic', end - start)

        infeasibility = max(0.0, f[1])
        history.popcol('f0',f[0]); history.popcol('inf',infeasibility)

        change=np.linalg.norm(x-x_old)

        print("{}: {:.3f} {:.3f} {:.3f}".format(converged.iteration - 1, f[0], infeasibility, change))
#
        #update the approximations
        obj.setpoint(x, f[0], df[0], aux)
        con1.setpoint(x, f[1], df[1], aux)
#
        #update the subproblem
        start = time.time()
        subproblem.build(x,f,df)
        end = time.time()
        print('sub build', end - start)
#
        x_old[:]=x
#   
        start = time.time()
        #solve the subproblem
        if sub =='pdip':
            x[:] = pdip(subproblem)[0]
        elif sub =='t2dual':
            x[:] = t2dual(subproblem)[0]
        elif sub =='osqp':
            x[:] = osqp(subproblem)[0]
        end = time.time()
        print('subsolve',end - start)

        cnt=cnt+1
#
        if os.path.exists("exit.flg"):
            break
#
    open("exit.flg", 'w').close()

    print("\n")
#
    return history
#
if __name__ == "__main__":
    sub='t2dual'
    petsctopopt_t2r(sub)
