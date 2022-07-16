#
import time
import numpy as np
#
from sao.convergence_criteria import IterationCount
from sao.move_limits import Bounds, MoveLimit, MoveLimitFraction, AdaptiveMoveLimit
from sao.problems.subproblem_func import Subproblem as Subproblem_func
from sao.problems import Subproblem
from sao.solvers.primal_dual_interior_point import pdip, Pdipx
from sao.solvers.allmmadual import allmmadual
from sao.approximations import Taylor1
from problems.topology_optimization.compliance import ComplianceMBB
from sao.intervening_variables.mma import MMA02 as MMA
#
from sao.util.records import Records
from sao.function import Function
from sao.function2 import Function as Function2
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
class myLIN(Function):
#
    def intercurve(self, x):
#
        c_x = np.zeros_like(x)
#
        y=x
        dy=np.ones_like(x)
        ddy=np.zeros_like(x)
#
        return y, dy, ddy, c_x
#
class myMMA(Function2):
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
#   def intercurve(self, x):
#
#       c_x = np.zeros_like(x)
#       start = time.time()
#       y=np.where(self.dg_k < 0e0 , 1e0 / (x - self.L_k) , 1e0 / (self.U_k - x) )
#       end = time.time()
#       start = time.time()
#       dy=np.where(self.dg_k < 0e0 , -1e0 / ((x - self.L_k)**2e0) , 1e0 / ((self.U_k - x)**2e0 ))
#       end = time.time()
#       start = time.time()
#       ddy=np.where(self.dg_k < 0e0 , 2e0 / ((x - self.L_k)**3e0) , 2e0 / ((self.U_k - x)**3e0 ))
#       end = time.time()
#
#       return y, dy, ddy, c_x
#
    def intervene(self, x):
        return np.where(self.dg_k < 0e0 , 1e0 / (x - self.L_k) , 1e0 / (self.U_k - x) )
    def dintervene(self, x):
        return np.where(self.dg_k < 0e0 ,-1e0 / ((x - self.L_k)**2e0) , 1e0 / ((self.U_k - x)**2e0 ))
    def ddintervene(self, x):
        return np.where(self.dg_k < 0e0 , 2e0 / ((x - self.L_k)**3e0) , 2e0 / ((self.U_k - x)**3e0 ))
    def curvature(self, x):
        return np.zeros_like(x)
#
def compliance_mma(t):
#
    x0 = 0.5
    nelx = 100*4
    nely = 50*4
    problem = ComplianceMBB(nelx, nely, vf=x0)
    bounds = Bounds(xmin=0, xmax=1)
#
    n = problem.n
    x = problem.x0
    x1 = problem.x0
    x2 = problem.x0
    x3 = problem.x0
    f = problem.g(x)
    df = problem.dg(x)
#
    aux=[t]
#
    obj = myMMA('Compliance',problem.n)
    con1 = myMMA('Volume',problem.n)
#
    #instantiate subproblem
    funcs = [obj, con1]
    subproblem_func = Subproblem_func(funcs, limits=[bounds, MoveLimit(0.2)])
    subproblem = Subproblem(Taylor1(MMA()), limits=[bounds, MoveLimit(0.2)])
#
    #instantiate history
    history = Records(['f0','inf'])
#
#   converged = IterationCount(10)
#   while not converged:
#
    f = problem.g(x); df = problem.dg(x)
    infeasibility = max(0.0, f[1])
    history.popcol('f0',f[0]); history.popcol('inf',infeasibility)
#   print("{}: {:.3f} {:.3f}".format(converged.iteration - 1, f[0], infeasibility))
#
    #update the approximations
    obj.setpoint(x, f[0], df[0], aux)
    con1.setpoint(x, f[1], df[1], aux)
#
    #update the subproblem
    subproblem_func.build(x,f,df)
    subproblem.build(x,f,df)
#
    #solve the subproblem
    start = time.time()
    x1[:],k = pdip(subproblem_func, variables=Pdipx)
    end = time.time()
    print("func: %e %d"%(end-start,k) )
    start = time.time()
    x2[:],k = pdip(subproblem, variables=Pdipx)
    end = time.time()
    print("stan: %e %d"%(end-start,k) )
    print(np.linalg.norm(x2-x1))

    start = time.time()
    x3[:] = allmmadual(subproblem_func,funcs)[0]
    end = time.time()
    print("dual: %e %d"%(end-start,k) )
    print(np.linalg.norm(x3-x1))


    print("\n")
#
    return history
#
if __name__ == "__main__":
    compliance_mma(1./2.)
