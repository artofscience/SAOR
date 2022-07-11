#
import numpy as np
#
from problems.svanberg1987 import CantileverBeam
from sao.convergence_criteria import IterationCount
from sao.move_limits import Bounds, MoveLimit, MoveLimitFraction, AdaptiveMoveLimit
from sao.problems.subproblem_func import Subproblem
from sao.solvers.primal_dual_interior_point import pdip, Pdipx
from sao.solvers.t2dual import t2dual
#
from sao.util.records import Records
from sao.function import Function
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
class R(Function):
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
            if dg_k[i] < 0e0:
                y[i] = 1./x[i]
                dy[i] = -1e0/x[i]**2e0
                ddy[i] = 2e0/x[i]**3e0
            else: # does not happen
                print('ERROR')
#               y[i] = 1./x[i]
#               dy[i] = -1e0/x[i]**2e0
#               ddy[i] = 2e0/x[i]**3e0
            c_x[i] = 0e0
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
def cantilever_t2r(sub):
#
    #instantiate the problem instance
    problem = CantileverBeam()
#
    #bounds and move limit stuff
    bounds = Bounds(xmin=problem.x_min, xmax=problem.x_max)
    movelimit = MoveLimit(move_limit=0.2,dx=problem.x_max-problem.x_min)
#
    n = problem.n
    x = problem.x0
    f = problem.g(x)
    df = problem.dg(x)
#
    aux=[]
#
    obj = T2R('Weight',problem.n)
    con1 = T2R('Displacement',problem.n)
#
    #instantiate subproblem
    funcs = [obj, con1]
    subproblem = Subproblem(funcs, limits=[bounds, movelimit])
#
    #instantiate history
    history = Records(['f0','inf'])
#
    converged=IterationCount(7)
#
    cnt=0
    while not converged:
        f = problem.g(x); df = problem.dg(x)
        infeasibility = max(0.0, f[1])
        history.popcol('f0',f[0]); history.popcol('inf',infeasibility)
        print("{}: {:.3f} {:.3f}".format(converged.iteration - 1, f[0], infeasibility))
#
        #update the approximations
        obj.setpoint(x, f[0], df[0], aux)
        con1.setpoint(x, f[1], df[1], aux)
#
        #update the subproblem
        subproblem.build(x,f,df)
#
        #solve the subproblem
        if sub =='pdip':
            x[:] = pdip(subproblem, variables=Pdipx)[0]
        elif sub =='t2dual':
            x[:] = t2dual(subproblem)[0]

    print("\n")
#
    return history
#
if __name__ == "__main__":
    sub='pdip'
    cantilever_t2r(sub)
    sub='t2dual'
    cantilever_t2r(sub)
