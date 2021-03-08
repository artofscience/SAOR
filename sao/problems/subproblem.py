from sao.problems.problem import Problem
from sao.approximations.intervening import Linear
from sao.approximations.taylor import Taylor1
from sao.move_limits.ml_intervening import MoveLimitIntervening


class Subproblem(Problem):
    def __init__(self, intervening=Linear(), approximation=Taylor1(), ml=MoveLimitIntervening()):
        super().__init__()
        self.inter = intervening
        self.approx = approximation
        self.ml = ml
        self.f, self.df, self.ddf = None, None, None
        self.alpha, self.beta = None, None

    def build(self, x, f, df, ddf=None):
        self.x = x
        self.f, self.df, self.ddf = f, df, ddf
        self.n, self.m = len(x), len(f) - 1             # to fit Stijn's solvers

        self.inter.update(x=self.x, f=self.f, df=self.df, ddf=self.ddf, xmin=self.ml.xmin, xmax=self.ml.xmax)

        # If available, handle 2nd-order information
        if self.ddf is not None:
            self.approx.update(self.inter.y(self.x).T, self.f, self.df * self.inter.dxdy(self.x),
                               self.ddf * (self.inter.dxdy(self.x)) ** 2 + self.df * (self.inter.ddxddy(self.x)))
        else:
            self.approx.update(self.inter.y(self.x).T, self.f, self.df * self.inter.dxdy(self.x))

        self.alpha, self.beta = self.ml.update(self.x, intervening=self.inter)

    def g(self, x):
        return self.approx.g(y=self.inter.y(x).T)

    def dg(self, x):
        return self.approx.dg(y=self.inter.y(x).T, dy=self.inter.dy(x))

    def ddg(self, x):
        return self.approx.ddg(y=self.inter.y(x).T, dy=self.inter.dy(x), ddy=self.inter.ddy(x))

    '''
    P = dg_j/dy_ji = dg_j/dx_i * dx_i/dy_ji [(m+1) x n]
    Q = d^2g_j/dy_ji^2 = d^2g_j/dx_i^2 * (dx_i/dy_ji)^2 + dg_j/dx_i * d^2x_i/dy_ji^2 [(m+1) x n]
    y = [(m+1) x n] or [n], depending on the intervening variables used (see ReferenceFiles/TaylorExpansion.pdf)
    x = [n]
        
          |   resp [1, 2, 3]  |     resp [4, 5]
    -------------------------------------------------         
    0-N   |  Taylor1 + MMA    |  Taylor1 + Conlin
    -------------------------------------------------    
    N-N+2 |  Taylor1 + linear |  Taylor1 + reciprocal
    
    
    P = [P1, P2]
        [P3, P4]
        
    Q = [0, 0 ]
        [0, Q4]
    
    '''
