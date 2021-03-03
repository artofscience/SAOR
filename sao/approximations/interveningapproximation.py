from .approximation import Approximation
from sao.approximations.intervening.intervening import Linear
from .taylor import Taylor1
from .bounds import Bounds


class InterveningApproximation(Approximation):
    def __init__(self, intervening=Linear(), approximation=Taylor1(), bounds=Bounds()):
        super().__init__()
        self.inter = intervening
        self.approx = approximation
        self.bounds = bounds

    def build_approximation(self):

        self.inter.update_intervening(x=self.x, g=self.f, dg=self.df, ddg=self.ddf,
                                      xmin=self.bounds.xmin, xmax=self.bounds.xmax)

        # Preserve convexity
        Q = self.ddf * (self.inter.dxdy(self.x)) ** 2 + self.df * (self.inter.ddxddy(self.x))
        Q[Q < 0] = 0

        self.approx.update_approximation(self.inter.y(self.x).T, self.f, self.df*self.inter.dxdy(self.x), Q)
        self.bounds.update_bounds(self.inter, self.x)

        # P = dg_j/dy_ji = dg_j/dx_i * dx_i/dy_ji [m x n]
        # Q = d^2g_j/dy_ji^2 = d^2g_j/dx_i^2 * (dx_i/dy_ji)^2 + dg_j/dx_i * d^2x_i/dy_ji^2 [m x n]
        # y = [m x n] or [n], depending on the intervening variables used (see ReferenceFiles/TaylorExpansion.pdf)
        # x = [n]

    def g(self, x):
        return self.approx.g(self.inter.y(x).T)

    def dg(self, x):
        return self.approx.dg(self.inter.y(x).T)

    def ddg(self, x):
        return self.approx.ddg(self.inter.y(x).T)


    '''
      resp [1, 2, 3]        resp [4, 5]
0-N   Taylor1 + MMA    |  Taylor1 + Conlin
    
N-N+2 Taylor1 + linear |  Taylor1 + reciprocal
    
    P = [P1, P2]
        [P3, P4]
        
    Q = [0, 0 ]
        [0, Q4]
    
    '''