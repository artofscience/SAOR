from .approximation import Approximation
from sao.approximations.intervening.intervening import Linear
from .taylor import Taylor1


# TODO: This name confused me a bit. What is an InterveningApproximation? Is it a 2nd (less abstract) layer of the
#  abstract Approximation? If there is no other use of intervening variables apart from a Taylor expansion, we might
#  consider merging this class with Taylor1 of taylor.py. Might simplify things somewhat.
class InterveningApproximation(Approximation):
    def __init__(self, intervening=Linear(), approximation=Taylor1()):
        super().__init__()
        self.inter = intervening
        self.approx = approximation

    def build_approximation(self):
        self.inter.update_intervening(x=self.x, f=self.f, df=self.df, ddf=self.ddf, xmin=self.xmin, xmax=self.xmax)
        self.approx.update_approximation(self.inter.y(self.x), self.f, self.df*self.inter.dy(self.x), '''TODO ddy''', )

        # P = df/dy_i = df/dx_i * dx_i/dy_i [m x n]
        # y = [n]
        # x = [n]

    def g_approx(self, x):
        return self.approx.g_approx(self.inter.y(x))

    def dg_approx(self, x):
        return self.approx.dg_approx(self.inter.y(x))

    def ddg_approx(self, x):
        return self.approx.ddg_approx(self.inter.y(x))


    '''
      resp [1, 2, 3]        resp [4, 5]
0-N   Taylor1 + MMA    |  Taylor1 + Conlin
    
N-N+2 Taylor1 + linear |  Taylor1 + reciprocal
    
    P = [P1, P2]
        [P3, P4]
        
    Q = [0, 0 ]
        [0, Q4]
    
    '''