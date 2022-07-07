#
import numpy as np
from scipy.optimize import minimize
#
def allcondual(problem):
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
    bds=[[0e0,1e8] for i in range(m)]; tup_bds=tuple(bds)
    sol=minimize(con_dual,x_d,args=(n,m,x_k,g,dg,x_l,x_u), \
        jac=dcon_dual,method='L-BFGS-B',bounds=tup_bds, options={'disp':False})
#
    x_d[:]=sol.x
    x=x_dual(x_d, n, m, x_k, g, dg, x_l, x_u)
    problem.x_d_k = x_d
#
    return x,x_d
#
# CONLIN: x in terms of dual variables 
#
def x_dual(x_d, n, m, x_k, g, dg, dx_l, dx_u):
#
    x = np.zeros(n,dtype=np.float64)
#
    tmpn = np.zeros(n,dtype=np.float64)
    tmpp = np.zeros(n,dtype=np.float64)
#
    for j in range(n):
        if dg[0][j] > 0e0:
            tmpp[j] = tmpp[j] + dg[0][j]
        else:
            tmpn[j] = tmpn[j] - dg[0][j]#*x_k[j]**2e0
        for i in range(m):
            if dg[i+1][j] > 0e0:
                tmpp[j] = tmpp[j] + dg[i+1][j]*x_d[i]
            else:
                tmpn[j] = tmpn[j] - dg[i+1][j]*x_d[i]#*x_k[j]**2e0
        tmpp[j]=max(tmpp[j],1e-6)
        tmpn[j]=max(tmpn[j],0e0)
#
    for j in range(n):
        x[j] = min(max(np.sqrt(tmpn[j]/tmpp[j])*x_k[j],dx_l[j]),dx_u[j])
#
    return x
#
# CONLIN: Dual function value
#
def con_dual(x_d, n, m, x_k, g, dg, dx_l, dx_u):
#
    x=x_dual(x_d, n, m, x_k, g, dg, dx_l, dx_u)
#
    W = g[0]
    for i in range(m):
        W = W + x_d[i]*g[i+1]
    for j in range(n):
        if dg[0][j] > 0e0:
            W = W + dg[0][j]*(x[j]-x_k[j])
        else:
            W = W - dg[0][j]*(1e0/x[j]-1e0/x_k[j])*(x_k[j])**2e0
        for i in range(m):
            if dg[i+1][j] > 0e0:
                W = W + dg[i+1][j]*(x[j]-x_k[j])*x_d[i]
            else:
                W = W - dg[i+1][j]*(1e0/x[j]-1e0/x_k[j])*(x_d[i])*(x_k[j])**2e0
#
    return -W
#
# CONLIN: Dual gradient
#
def dcon_dual(x_d, n, m, x_k, g, dg, dx_l, dx_u):
#
    x=x_dual(x_d, n, m, x_k, g, dg, dx_l, dx_u)
#
    dW = np.zeros(m,dtype=np.float64)
#
    for i in range(m):
        dW[i] = dW[i] + g[i+1]
        for j in range(n):
            if dg[i+1][j] > 0e0:
                dW[i] = dW[i] + dg[i+1][j]*(x[j]-x_k[j])
            else:
                dW[i] = dW[i] - dg[i+1][j]*(1e0/x[j]-1e0/x_k[j])*(x_k[j])**2e0
#
    return -dW
#
