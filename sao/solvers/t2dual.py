#
import numpy as np
from scipy.optimize import minimize
#
def t2dual(problem):
#
    n = problem.n
    m = problem.m
    x_k = problem.x_k
    x_d = problem.x_d_k
    x_l = problem.x_min 
    x_u = problem.x_max 
#
    g = problem.g(x_k)
    dg =problem.dg(x_k)
    ddg =problem.ddg(x_k)
#
    dx_l=np.ones(n,dtype=np.float64)
    dx_u=np.ones(n,dtype=np.float64)
#
    if np.any(ddg[0]<1e-6):
        print("WARNING; strictly convex Lagrangian required for definition of Falk dual")
        print("Setting objective curvatures to small, positive number")
        ddg[0]=ddg[0]+1e-6
#
    bds=[[0e0,1e8] for i in range(m)]; tup_bds=tuple(bds)
    sol=minimize(qpq_dual,x_d,args=(n,m,x_k,g,dg,x_l,x_u, ddg[0], ddg[1:]), \
        jac=dqpq_dual,method='L-BFGS-B',bounds=tup_bds, options={'disp':False})
#
    if sol.status != 0 or sol.success != True : print('Warning; subproblem')
#
    x_d[:]=sol.x
#
    x=x_dual(x_d, n, m, x_k, g, dg, x_l, x_u, ddg[0], ddg[1:])
#
    problem.x_d_k = x_d
#
    return x,x_d
#
# QPQC: x in terms of dual variables 
#
def x_dual(x_d, n, m, x_k, g, dg, dx_l, dx_u, c0, cj):
#
    ddL=(c0 + np.dot(x_d,cj))
    tmp=(dg[0]+np.dot(x_d,dg[1:]))
#
    return np.maximum(np.minimum(x_k - tmp/ddL, dx_u),dx_l)
#
# QPQC: Dual function value
#
def qpq_dual(x_d, n, m, x_k, g, dg, dx_l, dx_u, c0, cj):
#
    x=x_dual(x_d, n, m, x_k, g, dg, dx_l, dx_u, c0, cj)
#
    ddL=(c0 + np.dot(x_d,cj))
#
    W = g[0]+np.dot(dg[0],(x-x_k))+np.dot(ddL/2e0,(x-x_k)**2e0)+np.dot(x_d,(g[1:]+np.dot(dg[1:],(x-x_k))))
#
    return -W
#
# QPQC: Dual gradient
#
def dqpq_dual(x_d, n, m, x_k, g, dg, dx_l, dx_u, c0, cj):
#
    x=x_dual(x_d, n, m, x_k, g, dg, dx_l, dx_u, c0, cj)
#
    dW=g[1:]+np.dot(dg[1:],(x-x_k)) + np.dot(cj/2e0,(x-x_k)**2e0)
#
    return -dW
#
