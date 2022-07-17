#
import numpy as np
#
from problems.haftka1991 import TenBarTruss
from sao.convergence_criteria import IterationCount
from sao.move_limits import Bounds, MoveLimit, MoveLimitFraction, AdaptiveMoveLimit
from sao.problems.subproblem_func import Subproblem
from sao.solvers.primal_dual_interior_point import pdip, Pdipx
from sao.solvers.t2dual import t2dual
#
from sao.util.records import Records
from sao.function2 import Function
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
#
class T2R(Function):
#
    def intervene(self,x):
        return x
    def dintervene(self,x):
        return np.ones_like(x)
    def ddintervene(self,x):
        return np.zeros_like(x)
#
    def curvature(self,x):
        c_x=2e0*np.absolute(self.dg_k)/self.x_k
        return c_x
#
class T2C(Function):
#
    def intervene(self,x):
        return x
    def dintervene(self,x):
        return np.ones_like(x)
    def ddintervene(self,x):
        return np.zeros_like(x)
#
    def curvature(self,x):
        c_x=np.where(self.dg_k>0e0, -2e0*self.dg_k/self.x_k, 0e0)
        return c_x
#
class R(Function):
#
    def intervene(self,x):
        return 1./x
    def dintervene(self,x):
        return -1./x**2e0
    def ddintervene(self,x):
        return 2./x**3e0
    def curvature(self,x):
        c_x=np.zeros_like(x)
        return c_x
#
class L(Function):
#
    def intervene(self,x):
        return x
    def dintervene(self,x):
        return np.ones_like(x)
    def ddintervene(self,x):
        return np.zeros_like(x)
    def curvature(self,x):
        return np.ones_like(x)*1e-6
#
def tenbartruss_t2r(sub):
#
    #instantiate the problem instance
    problem = TenBarTruss()
#
    #bounds and move limit stuff
    bounds = Bounds(xmin=problem.x_min, xmax=problem.x_max)
    movelimit = MoveLimit(move_limit=1.0,dx=1e8)
#
    n = problem.n
    x = problem.x0
    f = problem.g(x)
    df = problem.dg(x)
#
    aux=[]
#
    funcs=[L('Weight',problem.n) ]
    for j in range(problem.m):
        funcs.append(T2R('Stress %d'%j,problem.n))
#
    #instantiate subproblem
    subproblem = Subproblem(funcs, limits=[bounds, movelimit])
#
    #instantiate history
    history = Records(['f0','inf'])
#
    converged=IterationCount(14)
#
    cnt=0
    while not converged:
        f = problem.g(x); df = problem.dg(x)
        infeasibility = max(0.0, np.max(f[1:]))
        history.popcol('f0',f[0]); history.popcol('inf',infeasibility)
        print("{}: {:.3f} {:.3f}".format(converged.iteration - 1, f[0], infeasibility))
#
        #update the approximations
        for j in range(problem.m+1):
            funcs[j].setpoint(x, f[j], df[j], aux, 3, True)
#
        #update the subproblem
        subproblem.build(x,f,df)
#
        #solve the subproblem
        if sub =='pdip': #seems to struggle with infeasibility
            x[:] = pdip(subproblem, variables=Pdipx)[0]
        elif sub =='t2dual': #see infeasibility in first subproblem; solution corresponds to Groen
            x[:] = t2dual(subproblem)[0]

    print("\n")
#
    return history
#
if __name__ == "__main__":
    sub='t2dual'
    tenbartruss_t2r(sub)
