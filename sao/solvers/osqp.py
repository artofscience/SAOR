#
import osqp as qp
import numpy as np
from scipy import sparse
#
def osqp(problem):
#
    n = problem.n
    m = problem.m
    x_k = problem.functions[0].x_k
    x_d_k = problem.x_d_k
    x_l = problem.x_min - x_k.transpose()
    x_u = problem.x_max - x_k.transpose()
#
    g = problem.g(x_k)
    dg =problem.dg(x_k)
    ddg =problem.ddg(x_k)
#
    c0=np.zeros(n,dtype=np.float64)
    cj=np.zeros((m,n),dtype=np.float64)
    ddL=np.zeros(n,dtype=np.float64)
    for i in range(n):
        c0[i]=ddg[0][i]
        for j in range(m):
            cj[j][i]=ddg[j+1][i]
            ddL[i]=ddL[i]+cj[j][i]*x_d_k[j]
        ddL[i]=ddL[i]+c0[i]
        ddL[i]=max(ddL[i],0e0)
#
    J=dg[0]; ind = np.array(range(n))
    Q=sparse.csc_matrix((ddL, (ind, ind)), shape=(n, n))
    tmp=np.zeros((n,n))
    np.fill_diagonal(tmp,1e0)
    A=sparse.csc_matrix(np.append(dg[1:],tmp,axis=0))
    u=-g[1:]; l=-np.ones(m,dtype=np.float64)*1e16
#
    l=np.append(l,x_l)
    u=np.append(u,x_u)
#
    prob = qp.OSQP()
    prob.setup(Q, J, A, l, u,verbose=False)
    res=prob.solve()
    if res.info.status_val != 1:  #https://github.com/osqp/osqp/blob/master/include/constants.h
#
        k=1e6
        J=np.append(dg[0],-k*np.ones(1))

        ind = np.array(range(n+1))
        ddL=np.append(ddL, np.zeros(1))
        Q=sparse.csc_matrix((ddL, (ind, ind)), shape=(n+1, n+1))
        
        tmp=np.zeros((n+1,n+1))
        np.fill_diagonal(tmp,1e0)
        A=sparse.csc_matrix(np.append(np.append(dg[1:],np.ones((m,1)),axis=1),tmp,axis=0))
        u=-g[1:]; l=-np.ones(m,dtype=np.float64)*1e16
        l=np.append(l,np.append(x_l,-1e8))
        u=np.append(u,np.append(x_u,0.0))
        infprob = qp.OSQP()
        infprob.setup(Q, J, A, l, u,verbose=False)
        res=infprob.solve()
        x_d=res.y[:m]
        x=x_k.transpose()+np.maximum(np.minimum(res.x[:n],x_u),x_l)
#
    else:
        x_d=res.y[:m]
#
        x=x_k.transpose()+np.maximum(np.minimum(res.x,x_u),x_l)
#
    problem.x_d_k = x_d
#
    return x,x_d
#
