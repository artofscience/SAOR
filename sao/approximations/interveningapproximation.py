from .approximation import Approximation
from sao.approximations.intervening.intervening import Linear
from .taylor import Taylor1


# TODO: This name confused me a bit. I am guessing this is the class that combine the Taylor expansion expressions
#  with the intervening vars. Also, this is the class from which the -approx- instance will be created.
class InterveningApproximation(Approximation):
    def __init__(self, intervening=Linear(), approximation=Taylor1()):
        super().__init__()
        self.inter = intervening
        self.approx = approximation

    # TODO: Not a fan of f, df, ddf, as I confuse them with g, dg, ddg. I would write g, dg, ddg for the exact values,
    #  and g_approx(x), dg_approx(x), ddg_approx(x) for the approximate functions.
    def build_approximation(self):
        self.inter.update_intervening(x=self.x, f=self.f, df=self.df, ddf=self.ddf, xmin=self.xmin, xmax=self.xmax)
        self.approx.update_approximation(self.inter.y(self.x), self.f, self.df*self.inter.dy(self.x),
                                         self.inter.dy(self.x), self.inter.ddy(self.x))

        # TODO: If we want to have y.shape = [n,] for the cases of ConLin and MMA (i.e. when y_i=T(x_i) depends on the
        #  sign of dg_j/dx_i), we must re-calculate y_i(x_i) for each response function g_j. Otherwise, we must generate
        #  an augmented y, for which y.shape = [n, m+1]. Then, each column of -y- corresponds to a response g_j.
        # P = df/dy_i = df/dx_i * dx_i/dy_i [m x n]
        # y = [n]
        # x = [n]

    # TODO: The zero-order terms are re-calculated every time g_approx is evaluated at a different point.
    #  This might occur tens of times within the solver. However, it is only necessary to compute the zero-order terms
    #  of g_approx once per design iteration.
    def g_approx(self, x):
        return self.approx.g_approx(self.inter.y(x))

    def dg_approx(self, x):
        return self.approx.dg_approx(self.inter.y(x))

    def ddg_approx(self, x):
        return self.approx.ddg_approx(self.inter.y(x))


    '''
      resp [1, 2, 3]        resp [4, 5]
0-N   Taylor1 + MMA    |  Taylor1 + ConLin
    
N-N+2 Taylor1 + linear |  Taylor1 + reciprocal
    
    P = [P1, P2]
        [P3, P4]
        
    Q = [0, 0 ]
        [0, Q4]
    '''