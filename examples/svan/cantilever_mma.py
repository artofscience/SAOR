#
import numpy as np
#
from problems.svanberg1987 import CantileverBeam
from sao.convergence_criteria import IterationCount
from sao.move_limits import Bounds, MoveLimit, MoveLimitFraction, AdaptiveMoveLimit
from sao.problems.subproblem_func import Subproblem
from sao.solvers.primal_dual_interior_point import pdip, Pdipx
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
class MMA(Function):
#
    def domain(self): #domain
#
        d_l = np.zeros(self.n,dtype=float)
        d_u = np.zeros(self.n,dtype=float)
        for i in range(self.n):
            d_l[i] = 1.01*self.L_k[i]
            d_u[i] = 0.99*self.U_k[i]
        return d_l, d_u
#
    def parameters(self,aux):
#
        x_k = self.x_k
        hst_x_k = self.hst_x_k
        k = self.k
#
        asy_fac = aux[0]
#
        L_k = np.zeros_like(x_k) 
        U_k = np.zeros_like(x_k) 
#
        for i in range(self.n):
            L_k[i]=asy_fac*x_k[i]
            U_k[i]=x_k[i]/asy_fac
#
        self.L_k = L_k
        self.U_k = U_k
#
    def intercurve(self, x):
#
        L_k = self.L_k
        U_k = self.U_k
#
        y = np.zeros_like(x)
        dy = np.zeros_like(x)
        ddy = np.zeros_like(x)
        c_x = np.zeros_like(x)
#
        x_k = self.x_k
        dg_k = self.dg_k
#
        for i in range(self.n):
            if dg_k[0][i] < 0e0:
                y[i] = 1e0 / (x[i] - L_k[i])
                dy[i] = -1e0 / (x[i] - L_k[i])**2e0
                ddy[i] = 2e0 / (x[i] - L_k[i])**3e0
            else:
                y[i] = 1e0 / (U_k[i] - x[i])
                dy[i] = 1e0 / (U_k[i] - x[i])**2e0
                ddy[i] = 2e0 / (U_k[i] - x[i])**3e0
#
        return y, dy, ddy, c_x
#
def cantilever_mma(t):
#
    #instantiate the problem instance
    problem = CantileverBeam()
#
    #bounds and move limit stuff
    bounds = Bounds(xmin=problem.x_min, xmax=problem.x_max)
    movelimit = MoveLimitFraction(fraction=2.0)
#
    n = problem.n
    x = problem.x0
    f = problem.g(x)
    df = problem.dg(x)
#
    aux=[t]
#
    obj = MMA('Weight',problem.n)
    con1 = MMA('Displacement',problem.n)
#
    #instantiate subproblem
    funcs = [obj, con1]
    subproblem = Subproblem(funcs, limits=[bounds, movelimit])
#
    #instantiate history
    history = Records(['f0','inf'])
#
    converged = IterationCount(15)
    cnt=0
    while not converged:
        f = problem.g(x); df = problem.dg(x)
        infeasibility = max(0.0, f[1])
        history.popcol('f0',f[0]); history.popcol('inf',infeasibility)
        print("{}: {:.3f} {:.3f}".format(converged.iteration - 1, f[0], infeasibility))
#
        if (infeasibility < 0.001) and (f[0] < 1.001 * problem.f_opt):
            break
#
        #update the approximations
        obj.setpoint(x, f[0], df[0], aux)
        con1.setpoint(x, f[1], df[1], aux)
#
        #update the subproblem
        subproblem.build(x,f,df)
#
        #solve the subproblem
        x[:] = pdip(subproblem, variables=Pdipx)[0]

    print("\n")
#
    return history
#
if __name__ == "__main__":
    cantilever_mma(1./2.)
