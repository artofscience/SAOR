from sao.problems.problem import Problem
import numpy as np

class TwoBarTruss(Problem):
    '''
    1. One element sizing variable x1 and one configuration variable x2
    2. One load case: external force F = [Fx, Fy] at the unsupported node (node 3)
    3. Fy = 8 Fx and F = 200 kN
    4. x1: cross-sectional area (cm2) of the bars
    5. x2: half of the distance (m) between node 1 and 2
    6. None of the variable bounds are active at the optimal solution
    7. The objective function is the weight of the bars
    8. The tensile stress must not be greater than 100 N/mm2 in either of the two bars

    '''

    def __init__(self):
        super().__init__()
        self.n = 2
        self.m = 2
        self.x_min = np.array([0.2, 0.1])
        self.x_max = np.array([4.0, 1.6])
        self.x0 = np.array([1.5, 0.5])
        self.name = 'TwoBarTruss'
        self.c2 = 0.124

    def g(self, x):
        g = np.zeros((self.m+1), dtype=float)
        tmp1 = np.sqrt(1 + x[1]**2)
        tmp2 = 8/x[0] + 1/(x[0]*x[1])
        tmp3 = 8/x[0] - 1/(x[0]*x[1])

        # Weight of the structure w[x1, x2]
        g[0] = x[0]*tmp1
        # Stress of bar 1 sigma1[x1, x2]
        g[1] = self.c2*tmp1*tmp2 - 1
        # Stress of bar 1 sigma1[x1, x2]
        g[2] = self.c2*tmp1*tmp3 - 1

        return g

    def dg(self, x):
        dg = np.zeros((self.m+1, self.n), dtype=float)
        tmp1 = np.sqrt(1 + x[1]**2)
        tmp2 = 8/x[0] + 1/(x[0]*x[1])
        tmp3 = 8/x[0] - 1/(x[0]*x[1])
        tmp4 = 2*x[1]

        dg[0][0] = tmp1
        dg[0][1] = x[0]/(2*tmp1)*tmp4
        dg[1][0] = -self.c2*tmp1*(8/x[0]**2 + 1/(x[0]**2*x[1]))
        dg[1][1] = self.c2/(2*tmp1)*tmp4*tmp2 - self.c2*tmp1/(x[0]*x[1]**2)
        dg[2][0] = -self.c2*tmp1*(8/x[0]**2 - 1/(x[0]**2*x[1]))
        dg[2][1] = self.c2/(2*tmp1)*tmp4*tmp3 + self.c2*tmp1/(x[0]*x[1]**2)

        return dg

if __name__ == "__main__":
    prob = TwoBarTruss()
    x = (prob.x_min + prob.x_max)/2
    g0 = prob.g(x)
    dg_an = prob.dg(x)

    dx = 1e-7
    dg_fd = np.zeros_like(dg_an)
    for i in range(prob.n):
        x0 = x[i]
        x[i] += dx
        gp = prob.g(x)
        x[i] = x0
        dg_fd[:, i] = (gp - g0) / dx
        print(f"an: {dg_an[:, i]}, fd: {dg_fd[:, i]}, diff = {dg_an[:, i]/dg_fd[:, i] - 1.0}")


