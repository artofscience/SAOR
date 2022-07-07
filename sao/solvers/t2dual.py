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
    x = np.zeros(n,dtype=np.float64)
#
    ddL=np.zeros(n,dtype=np.float64)
    for i in range(n):
        ddL[i]=ddL[i]+c0[i]
        for j in range(m):
            ddL[i]=ddL[i]+cj[j][i]*x_d[j]
#
    tmp=np.zeros(n,dtype=np.float64)
    for i in range(n):
        tmp[i]=tmp[i]+dg[0][i]
        for j in range(m):
            tmp[i]=tmp[i]+x_d[j]*dg[j+1][i]
    for i in range(n):
        x[i] = max(min(x_k[i] - tmp[i]/ddL[i],dx_u[i]),dx_l[i])
#
    return x
#
# QPQC: Dual function value
#
def qpq_dual(x_d, n, m, x_k, g, dg, dx_l, dx_u, c0, cj):
#
    x=x_dual(x_d, n, m, x_k, g, dg, dx_l, dx_u, c0, cj)
#
    ddL=np.zeros(n,dtype=np.float64)
    for i in range(n):
        ddL[i]=ddL[i]+c0[i]
        for j in range(m):
            ddL[i]=ddL[i]+cj[j][i]*x_d[j]
#
    W = g[0]
    for i in range(n):
        W = W + dg[0][i]*(x[i]-x_k[i]) + ddL[i]/2e0*(x[i]-x_k[i])**2e0
    for j in range(m):
        W = W + x_d[j]*g[j+1]
        for i in range(n): 
            W = W + x_d[j]*dg[j+1][i]*(x[i]-x_k[i])
#
    return -W
#
# QPQC: Dual gradient
#
def dqpq_dual(x_d, n, m, x_k, g, dg, dx_l, dx_u, c0, cj):
#
    x=x_dual(x_d, n, m, x_k, g, dg, dx_l, dx_u, c0, cj)
#
    dW = np.zeros(m,dtype=np.float64)
#
    for j in range(m):
        dW[j] = dW[j] + g[j+1]
        for i in range(n):
            dW[j] = dW[j] + dg[j+1][i]*(x[i]-x_k[i]) + cj[j][i]/2e0*(x[i]-x_k[i])**2e0
#
    return -dW
#
