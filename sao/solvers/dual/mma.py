import numpy as np
from scipy.optimize import minimize
from sao.approximations.taylor import Taylor1
from sao.intervening_variables import MixedIntervening
from sao.intervening_variables.mma import MMAp


def sub_mma(prob, x, y):
    """
    MMA DUAL SOLVER

    :param x: current design variables (primal variables)
    :param y: current lagrange multipliers (dual variables)
    :return:
    """
    assert isinstance(prob.approx, Taylor1)
    if isinstance(prob.approx.interv[0], MixedIntervening):
        for y_of_x in prob.approx.interv[0].iv_mapping:
            assert isinstance(y_of_x[0], MMAp)
        # CHECK FOR OVERLAP BETWEEN RESPONSES
        # MULTIPLE ASYMPTOTES PER VARIABLE NOT ALLOWED
        for i in range(0, prob.m+1):
            for y_of_x in prob.approx.interv[0].iv_mapping:
                pass # SET ASMPTOTE VALUES
    else:
        assert isinstance(prob.approx.interv[0], MMAp)
        L = prob.approx.interv[0].low
        U = prob.approx.interv[0].upp

    g = prob.g(x)
    dg = prob.dg(x)

    r = np.zeros((prob.m+1),dtype=np.float64)
    p = np.zeros((prob.m+1,prob.n),dtype=np.float64)
    q = np.zeros((prob.m+1,prob.n),dtype=np.float64)

    for i in range(prob.m+1):
        r[i] = g[i]
        for j in range(prob.n):
            if dg[i][j] > 0e0:
                p[i][j] = dg[i][j]*(U[j]-x[j])**2e0;
                q[i][j] = 0e0
            else:
                q[i][j] = -dg[i][j]*(x[j]-L[j])**2e0;
                p[i][j] = 0e0
            r[i] = r[i] - p[i][j]/(U[j]-x[j]) - q[i][j]/(x[j]-L[j])

    sol = minimize(mma_dual,y,
                   args=(prob.n,prob.m,r,p,q,prob.x_min,prob.x_max,L,U),
                   jac=dmma_dual,
                   method='L-BFGS-B',
                   bounds=tuple([[0e0,1e8] for i in range(prob.m)]),
                   options={'disp':False})
    y[:] = sol.x
    x = x_dual(y, prob.n, prob.m, r, p, q, prob.x_min, prob.x_max, L, U)
    return [x, y]

# MMA: x in terms of dual variables
def x_dual(x_d, n, m, r, p, q, dx_l, dx_u, L, U):
    x = np.zeros(n,dtype=np.float64)
    tmp1 = np.zeros(n,dtype=np.float64)
    tmp2 = np.zeros(n,dtype=np.float64)
    for j in range(n):
        tmp1[j]=p[0][j]; tmp2[j]=q[0][j]
        for i in range(m):
            tmp1[j]=tmp1[j]+p[i+1][j]*x_d[i]; tmp2[j]=tmp2[j]+q[i+1][j]*x_d[i]
        tmp1[j]=max(tmp1[j],0e0); tmp2[j]=max(tmp2[j],0e0)
    for j in range(n):
        x[j] = (np.sqrt(tmp1[j])*L[j]+np.sqrt(tmp2[j])*U[j])/(np.sqrt(tmp1[j])+np.sqrt(tmp2[j]))
        x[j] = min(max(x[j],dx_l[j]),dx_u[j])
    return x

# MMA: Dual function value
def mma_dual(x_d, n, m, r, p, q, dx_l, dx_u, L, U):
    x=x_dual(x_d, n, m, r, p, q, dx_l, dx_u, L, U)
    tmp11=0e0; tmp22=0e0
    for j in range(n):
        tmp11=tmp11+p[0][j]/(U[j]-x[j]); tmp22=tmp22+q[0][j]/(x[j]-L[j])
        for i in range(m):
            tmp11=tmp11+p[i+1][j]*x_d[i]/(U[j]-x[j]); tmp22=tmp22+q[i+1][j]*x_d[i]/(x[j]-L[j])
    W = r[0] + tmp11 + tmp22
    for i in range(m):
        W = W - x_d[i]*(0e0-r[i+1])
    return -W

# MMA: Dual gradient
def dmma_dual(x_d, n, m, r, p, q, dx_l, dx_u, L, U):
    x=x_dual(x_d, n, m, r, p, q, dx_l, dx_u, L, U)
    tmp11=np.zeros(m); tmp22=np.zeros(m)
    for i in range(m):
        for j in range(n):
            tmp11[i]=tmp11[i]+p[i+1][j]/(U[j]-x[j]); tmp22[i]=tmp22[i]+q[i+1][j]/(x[j]-L[j])
    dW = np.zeros(m,dtype=np.float64)
    for i in range(m):
        dW[i] = dW[i] -(0e0-r[i+1]) + tmp11[i] + tmp22[i]
    return -dW