#
import numpy as np
from scipy.optimize import minimize
#
def allmmadual(problem, funcs):
#
    n = problem.n
    m = problem.m
    x_k = problem.x_k
    x_d = problem.x_d_k
    x_l = problem.x_min 
    x_u = problem.x_max 
#
    L = -np.ones(n)*1e8
    U = np.ones(n)*1e8
    for j in range(m+1):
        for i in range(n):
            L[i] = max(funcs[j].L_k[i],L[i])
            U[i] = min(funcs[j].U_k[i],U[i])
#
    g = problem.g(x_k)
    dg =problem.dg(x_k)
    ddg =problem.ddg(x_k)
#
    r = np.zeros((m+1),dtype=np.float64)
    p = np.zeros((m+1,n),dtype=np.float64)
    q = np.zeros((m+1,n),dtype=np.float64)
    for i in range(m+1):
        r[i] = g[i]
        for j in range(n):
            if dg[i][j] > 0e0:
                p[i][j] = dg[i][j]*(U[j]-x_k[j])**2e0; q[i][j] = 0e0
            else:
                q[i][j] = -dg[i][j]*(x_k[j]-L[j])**2e0; p[i][j] = 0e0
            r[i] = r[i] - p[i][j]/(U[j]-x_k[j]) - q[i][j]/(x_k[j]-L[j])
#
    bds=[[0e0,1e8] for i in range(m)]; tup_bds=tuple(bds)
    sol=minimize(mma_dual,x_d,args=(n,m,r,p,q,x_l,x_u,L,U), \
        jac=dmma_dual,method='L-BFGS-B',bounds=tup_bds, options={'disp':False})
#
    x_d[:]=sol.x
    x=x_dual(x_d, n, m, r, p, q, x_l, x_u, L, U)
#
    problem.x_d_k = x_d
#
    return x,x_d
#
# MMA: x in terms of dual variables 
#
def x_dual(x_d, n, m, r, p, q, dx_l, dx_u, L, U):
#
    x = np.zeros(n,dtype=np.float64)
    tmp1 = np.zeros(n,dtype=np.float64)
    tmp2 = np.zeros(n,dtype=np.float64)
#
    for j in range(n):
        tmp1[j]=p[0][j]; tmp2[j]=q[0][j]
        for i in range(m):
            tmp1[j]=tmp1[j]+p[i+1][j]*x_d[i]; tmp2[j]=tmp2[j]+q[i+1][j]*x_d[i]
        tmp1[j]=max(tmp1[j],0e0); tmp2[j]=max(tmp2[j],0e0)
#
    for j in range(n):
        x[j] = (np.sqrt(tmp1[j])*L[j]+np.sqrt(tmp2[j])*U[j])/(np.sqrt(tmp1[j])+np.sqrt(tmp2[j]))
        x[j] = min(max(x[j],dx_l[j]),dx_u[j])
#
    return x
#
# MMA: Dual function value
#
def mma_dual(x_d, n, m, r, p, q, dx_l, dx_u, L, U):
#
    x=x_dual(x_d, n, m, r, p, q, dx_l, dx_u, L, U)
#
    tmp11=0e0; tmp22=0e0
    for j in range(n):
        tmp11=tmp11+p[0][j]/(U[j]-x[j]); tmp22=tmp22+q[0][j]/(x[j]-L[j])
        for i in range(m):
            tmp11=tmp11+p[i+1][j]*x_d[i]/(U[j]-x[j]); tmp22=tmp22+q[i+1][j]*x_d[i]/(x[j]-L[j])
#
    W = r[0] + tmp11 + tmp22
    for i in range(m):
        W = W - x_d[i]*(0e0-r[i+1])
#
    return -W
#
# MMA: Dual gradient
#
def dmma_dual(x_d, n, m, r, p, q, dx_l, dx_u, L, U):
#
    x=x_dual(x_d, n, m, r, p, q, dx_l, dx_u, L, U)
#
    tmp11=np.zeros(m); tmp22=np.zeros(m)
    for i in range(m):
        for j in range(n):
            tmp11[i]=tmp11[i]+p[i+1][j]/(U[j]-x[j]); tmp22[i]=tmp22[i]+q[i+1][j]/(x[j]-L[j])
#
    dW = np.zeros(m,dtype=np.float64)
    for i in range(m):
        dW[i] = dW[i] -(0e0-r[i+1]) + tmp11[i] + tmp22[i]
#
    return -dW
#
