#
import numpy as np
from sao.problems.problem import Problem
#
class TenBarTruss(Problem):
#
#   haftkas 10-bar-truss (used by Haftka and Groen)
#   formulated in line with van Keulen truss element examples in Continuum reader
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
        L=360
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
#       area and younds modulis (element props)
        A = x; E = np.ones_like(A)
#
#       global load vector
        F = np.zeros((2*n_nds,1))
        F[3]=-100
        F[7]=-100
#
#       fixed displacement boundary conditions
        B = np.zeros(4,dtype=int)
        B[0]=8
        B[1]=9
        B[2]=10
        B[3]=11
#
        S={}; L={}; D={}; T={}
        K=np.zeros((2*n_nds,2*n_nds),dtype=np.float64)
        sig=np.zeros(n_elm,dtype=np.float64)
        eps=np.zeros(n_elm,dtype=np.float64)
#
#       assembly
        for e in range(n_elm):
#
#           length
            L[e] = np.linalg.norm(n_cor[e_con[e][1]]- n_cor[e_con[e][0]])
#
#           d_loc=[u_2_x, u_2_y, u_1_x, u_1_y]^T
#
#           D matrix (gen def given disp. in local system)
            D[e] = np.array([[1., 0., -1., 0.]])#/L[e]
#
#           transformation matrix (global to local)
            cos = (n_cor[e_con[e][1]][0]- n_cor[e_con[e][0]][0])/L[e]
            sin = (n_cor[e_con[e][1]][1]- n_cor[e_con[e][0]][1])/L[e]
            T[e] = np.array([[cos, sin, 0, 0],[-sin, cos, 0, 0],[0,0,cos,sin],[0.,0.,-sin,cos]])
#
#           material law (spring stiffness, given definition of gen def)
            S[e] = A[e]*E[e]/L[e]
#
#           element stiffness matrix, transformed to global
            tmp=np.dot(D[e],T[e])
            k = S[e]*np.dot(tmp.transpose(), tmp)
#
#           assemble
            ids = np.array([2*e_con[e][1],2*e_con[e][1]+1,2*e_con[e][0],2*e_con[e][0]+1])
            K[ids[:,np.newaxis],ids]+=k
#
#       remove fixed (at zero) dofs
        tmp = np.delete(K,list(B),0)
        K0 = np.delete(tmp,list(B),1)
        F0 = np.delete(F,list(B))
#
#       solve at free dofs for loads
        U0 = np.dot(np.linalg.inv(K0),F0)
#
#       pack into complete dof array
        U = np.zeros(2*n_nds)
        free=list(set(range(2*n_nds))-set(B))
        U[free]=U0
#
#       post process
        for e in range(n_elm):
            ids = np.array([2*e_con[e][1],2*e_con[e][1]+1,2*e_con[e][0],2*e_con[e][0]+1])
            dglo=U[ids]
            dloc= np.dot(T[e],dglo)
#           gen def
            eps[e]= np.dot(D[e],dloc)
#           gen stress (WHICH IS A LOAD!!!)
            sig[e]=  S[e]*eps[e]
#
#       make response functions
        for e in range(n_elm):
            g[0]=g[0]+A[e]*L[e]*0.1
            g[e+1] = sig[e]/A[e] - 25e0
            g[e+1+n_elm] = -sig[e]/A[e] - 25e0
            if e == 8:
                g[e+1] = sig[e]/A[e] - 75e0
                g[e+1+n_elm] = -sig[e]/A[e] - 75e0
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
