import numpy as np

from sao.problems.problem import Problem


class CantileverBeam(Problem):
    """
    1. Consider a cantilever beam built from n beam elements.
    2. Each beam element has a quadratic cross-section.
    3. The beam is rigidly supported at node 1.
    4. A given external vertical force is acting at node 6.
    5. The design variables are the heights of the beam elements
    6. The thicknesses are held fixed.
    7. The objective is the weight of the beam.
    8. The constraint is the tip displacement.
    9. The variable bounds are so small/large that they never become active.
    """

    def __init__(self):
        super().__init__()
        self.n = 5
        self.m = 1
        self.x0 = 5.0 * np.ones((self.n), dtype=float)
        self.x_min = 1.0 * np.ones_like(self.x0)
        self.x_max = 10.0 * np.ones_like(self.x0)
        self.name = 'CantileverBeam'
        self.c1 = 0.0624
        self.c2 = np.array([61, 37, 19, 7, 1], dtype=float)
        self.x_opt = np.array([6.016, 5.309, 4.494, 3.502, 2.153])
        self.f_opt = 1.340

    def g(self, x):
        g = np.zeros((self.m + 1), dtype=float)
        g[0] = self.c1 * np.sum(x)
        g[1] = np.dot(self.c2, 1 / x ** 3) - 1
        return g

    def dg(self, x):
        dg = np.zeros((self.m + 1, self.n), dtype=float)
        dg[0][:] = self.c1
        dg[1][:] = -3 * self.c2 / x ** 4
        return dg

    def ddg(self, x):
        ddg = np.zeros((self.m + 1, self.n), dtype=float)
        ddg[0][:] = 0.0
        ddg[1][:] = 12 * self.c2 / x ** 5
        return ddg


class EigthBarTruss(Problem):
    """
    1. Consider a simple truss containing 8 elements (bars).
    2. There is one load case: an external force F = [Fx, Fy, Fz] at node 5.
    3. Fx = 40 kN, Fy = 20 kN, Fz = 200 kN.
    4. The design variables are the cross-sectional areas of the elements.
    5. Lowerbounds are 100 mm2, upperbound are inf (very large)
    6. The objective function is the weight of the structure.
    7. The only constraints are 8 stress constraints (100 N/mm2)
    """

    def __init__(self):
        super().__init__()
        self.n = 8
        self.m = 16
        self.x0 = 400 * np.ones((self.n), dtype=float)
        self.x_min = 100 * np.ones_like(self.x0)
        self.x_max = 1e8 * np.ones_like(self.x0)
        self.name = 'EightBarTruss'
        self.x_opt = np.array([880, 720, 260, 520, 100, 100, 100, 100], dtype=float)
        self.f_opt = 11.23

    def g(self, x):
#
        g = np.zeros((self.m + 1), dtype=float)
#
        n=self.n
        m=self.m
#
        x_p=x
#
        fx=40e3
        fy=20e3
        fz=200e3
        f=np.array([fx,fy,fz])
#
        c=np.zeros((n+1,3),dtype=np.float64)
        c[0]=np.array([-250.,-250.,0.]); c[1]=np.array([-250.,250.,0.])
        c[2]=np.array([250.,250.,0.]); c[3]=np.array([250.,-250.,0.])
        c[4]=np.array([0.,0.,375.]); c[5]=np.array([-375.,0.,0.])
        c[6]=np.array([0.,375.,0.]); c[7]=np.array([375.,0.,0.])
        c[8]=np.array([0.,-375.,0.])
#
        d={}
        d[0]=[0,4]; d[1]=[1,4]; d[2]=[2,4]; d[3]=[3,4]
        d[4]=[5,4]; d[5]=[6,4]; d[6]=[7,4]; d[7]=[8,4]
#
        L=np.zeros(n,dtype=np.float64)
        K=np.zeros((3,3),dtype=np.float64)
        for i in range(n):
#
            L[i]=np.linalg.norm(c[d[i]][1]-c[d[i][0]])
            cx = (c[d[i][0]][0] - c[d[i][1]][0])/L[i]
            cy = (c[d[i][0]][1] - c[d[i][1]][1])/L[i]
            cz = (c[d[i][0]][2] - c[d[i][1]][2])/L[i]
            K=K+x_p[i]/L[i]*np.array([[cx**2.,cx*cy,cx*cz],[cx*cy,cy**2.,cy*cz],[cx*cz,cy*cz,cz**2.]])
#
        u=np.matmul(np.linalg.inv(K),f)
        sig=np.zeros(n,dtype=np.float64)
#
        for i in range(n):
#
            cx = (c[d[i][1]][0] - c[d[i][0]][0])/L[i]
            cy = (c[d[i][1]][1] - c[d[i][0]][1])/L[i]
            cz = (c[d[i][1]][2] - c[d[i][0]][2])/L[i]
#
            sig[i]= 1e0/L[i]*(   cx*u[0] + cy*u[1] + cz*u[2]   )
#
        g=np.zeros((1+m),dtype=np.float64)
#
        for i in range(8):
            g[0]=g[0]+x_p[i]*L[i]/128211.
            g[i+1] = sig[i] -100e0
            g[i+1+8] = -sig[i] -100e0
#
        return g

    def dg(self, x):
#
        from problems.util.fd import finite_difference_use
#
        dg = np.zeros((self.m + 1, self.n), dtype=float)
#
        dx = 1e-4
        problem = EigthBarTruss()
        dg[:] = finite_difference_use(problem, x, dx)
#
        return dg
#
class TwoBarTruss(Problem):
#
    '''
    1. One element sizing variable x1 and one configuration variable x2
    2. One load case: external force F = [Fx, Fy] at the free dofs (node 3)
    3. Fy = 8 Fx and F = 200 kN
    4. x1: cross-sectional area (cm2) of the bars
    5. x2: half of the distance (m) between node 1 and 2
    6. None of the variable bounds are active at the optimal solution
    7. The objective function is the weight of the bars
    8. The tensile stress must not be greater than 100 N/mm2 in either of the two bars

    '''
#
    def __init__(self):
        super().__init__()
        self.n = 2
        self.m = 2
        self.x_min = np.array([0.2, 0.1])
        self.x_max = np.array([4.0, 1.6])
        self.x0 = np.array([1.5, 0.5])
        self.name = 'TwoBarTruss'
        self.c1 = 1.0
        self.c2 = 0.124
        self.f_opt = 1.51
#
    def g(self, x):
        g = np.zeros((self.m + 1), dtype=float)
        tmp1 = np.sqrt(1 + x[1] ** 2)
        tmp2 = 8 / x[0] + 1 / (x[0] * x[1])
        tmp3 = 8 / x[0] - 1 / (x[0] * x[1])
#
        # Weight of the structure w[x1, x2]
        g[0] = self.c1 * x[0] * tmp1
        # Stress of bar 1 sigma1[x1, x2]
        g[1] = self.c2 * tmp1 * tmp2 - 1
        # Stress of bar 1 sigma1[x1, x2]
        g[2] = self.c2 * tmp1 * tmp3 - 1
#
        return g
#
    def dg(self, x):
        dg = np.zeros((self.m + 1, self.n), dtype=float)
        tmp1 = np.sqrt(1 + x[1] ** 2)
        tmp2 = 8 / x[0] + 1 / (x[0] * x[1])
        tmp3 = 8 / x[0] - 1 / (x[0] * x[1])
        tmp4 = 2 * x[1]
#
        dg[0][0] = tmp1
        dg[0][1] = x[0] / (2 * tmp1) * tmp4
        dg[1][0] = -self.c2 * tmp1 * (8 / x[0] ** 2 + 1 / (x[0] ** 2 * x[1]))
        dg[1][1] = self.c2 / (2 * tmp1) * tmp4 * tmp2 - self.c2 * tmp1 / (x[0] * x[1] ** 2)
        dg[2][0] = -self.c2 * tmp1 * (8 / x[0] ** 2 - 1 / (x[0] ** 2 * x[1]))
        dg[2][1] = self.c2 / (2 * tmp1) * tmp4 * tmp3 + self.c2 * tmp1 / (x[0] * x[1] ** 2)
#
        return dg
#
if __name__ == "__main__":
#
    from problems.util.fd import finite_difference
#
    dx = 1e-7
    problem = TwoBarTruss()
    finite_difference(problem, problem.x0 + 1, dx)
#
    problem = CantileverBeam()
    finite_difference(problem, problem.x0, dx)
#
