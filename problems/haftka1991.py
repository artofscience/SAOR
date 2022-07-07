#
import numpy as np
from sao.problems.problem import Problem
#
class TenBarTruss(Problem):
#
    def __init__(self):
        super().__init__()
        self.n = 10
        self.m = 20
        self.x0 = 5.0 * np.ones((self.n), dtype=float)
        self.x_min = .1e0 * np.ones_like(self.x0)
        self.x_max = 15e0 * np.ones_like(self.x0)
        self.name = 'TenBarTruss'
        self.x_opt = np.array([7.9, 0.1, 8.1, 3.9, 0.1, 0.1, 5.8, 5.51, 3.68, 0.14], dtype=float)
        self.f_opt = 1514.

    def g(self, x):
#
        g = np.zeros((self.m + 1), dtype=float)
#
        x_p=x
#
        P=-100; L=360
#
        n_elm=10; n_nds=6; n_dim=2
#
#       nodal coordinates
        n_cor=np.zeros((n_nds,n_dim),dtype=np.float64)
        n_cor[0]=np.array([2*L,L])
        n_cor[1]=np.array([2*L,0.])
        n_cor[2]=np.array([L,L])
        n_cor[3]=np.array([L,0.])
        n_cor[4]=np.array([0.,L])
        n_cor[5]=np.array([0.,0.])
#
#       element connectivities
        e_con={}
        e_con[0]=[4,2]
        e_con[1]=[2,0]
        e_con[2]=[5,3]
        e_con[3]=[3,1]
        e_con[4]=[2,3]
        e_con[5]=[0,1]
        e_con[6]=[4,3]
        e_con[7]=[2,5]
        e_con[8]=[2,1]
        e_con[9]=[0,3]
#
#       area and younds modulis
        A = x_p; E = np.ones_like(A)
#
#       load vector
        F = np.zeros((2*n_nds,1))
        F[3]=-100
        F[7]=-100
#
#       boundary conditions
        B = np.zeros(4,dtype=int)
        B[0]=8
        B[1]=9
        B[2]=10
        B[3]=11
#
        S=np.zeros(n_elm,dtype=np.float64)
        L=np.zeros(n_elm,dtype=np.float64)
        K=np.zeros((2*n_nds,2*n_nds),dtype=np.float64)
#
        for e in range(n_elm):
#
            L[e] = np.linalg.norm(n_cor[e_con[e][1]]- n_cor[e_con[e][0]])
            lox = (n_cor[e_con[e][1]][0]- n_cor[e_con[e][0]][0])/L[e]
            mox = (n_cor[e_con[e][1]][1]- n_cor[e_con[e][0]][1])/L[e]
            Lam = np.array([[lox, mox, 0, 0],[0, 0, lox, mox]])
            k = np.array([[1, -1],[-1, 1]])
            k = k*A[e]*E[e]/L[e]
            klocal = np.dot(np.dot(Lam.transpose(), k), Lam)
#
            id1=2*e_con[e][0]
            id2=id1+1
            id3=2*e_con[e][1]
            id4 =id3+1
#
            K[id1,id1]+=klocal[0,0]
            K[id1,id2]+=klocal[0,1]
            K[id2,id1]+=klocal[1,0]
            K[id2,id2]+=klocal[1,1]

            K[id1,id3]+=klocal[0,2]
            K[id1,id4]+=klocal[0,3]
            K[id2,id3]+=klocal[1,2]
            K[id2,id4]+=klocal[1,3]

            K[id3,id1]+=klocal[2,0]
            K[id3,id2]+=klocal[2,1]
            K[id4,id1]+=klocal[3,0]
            K[id4,id2]+=klocal[3,1]

            K[id3,id3]+=klocal[2,2]
            K[id3,id4]+=klocal[2,3]
            K[id4,id3]+=klocal[3,2]
            K[id4,id4]+=klocal[3,3]
#
        tmp = np.delete(K,list(B),0)
        K0 = np.delete(tmp,list(B),1)
        F0 = np.delete(F,list(B))
#
        U0 = np.dot(np.linalg.inv(K0),F0)
#
        U = np.zeros((2*n_nds,1))
        j=0
        for i in range(2*n_nds):
            if i not in list(B):
                U[i]=U0[j]
            j=j+1
#
        for e in range(n_elm):
            lox = (n_cor[e_con[e][1]][0]- n_cor[e_con[e][0]][0])/L[e]
            mox = (n_cor[e_con[e][1]][1]- n_cor[e_con[e][0]][1])/L[e]
            id1=2*e_con[e][0]
            id2=id1+1
            id3=2*e_con[e][1]
            id4 =id3+1
            ulocal= np.array([U[id1],U[id2],U[id3],U[id4]])
            S[e]=E[e]/L[e]*np.dot(  np.array([-lox,-mox,lox,mox]), ulocal )
#
        for e in range(n_elm):
            g[0]=g[0]+x_p[e]*L[e]*0.1
            g[e+1] = S[e] - 25e0
            g[e+1+n_elm] = -S[e] - 25e0
            if e == 8:
                g[e+1] = S[e] - 75e0
                g[e+1+n_elm] = -S[e] - 75e0
#
        return g

    def dg(self, x):
#
        from problems.util.fd import finite_difference_use
#
        dg = np.zeros((self.m + 1, self.n), dtype=float)
#
        dx = 1e-6
        problem = TenBarTruss()
        dg[:] = finite_difference_use(problem, x, dx)
#
        return dg
#
if __name__ == "__main__":
#
    from problems.util.fd import finite_difference
#
    dx = 1e-7
    problem = TenBarTruss()
    x_opt = np.array([7.9, 0.1, 8.1, 3.9, 0.1, 0.1, 5.8, 5.51, 3.68, 0.14], dtype=float)
    print(problem.g(x_opt))
#
