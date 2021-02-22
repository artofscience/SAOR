from .approximation import Approximation


class Taylor1(Approximation):
    def g_approx(self, x):
        return self.f + self.df.dot(x-self.x)

    def dg_approx(self, x):
        return self.df

    def ddg_approx(self, x):
        return 1.0


class Taylor2(Taylor1):
    def g_approx(self, x):
        return super().g_approx(x) + 0.5*self.ddf.dot((x-self.x)**2)

    def dg_approx(self, x):
        return super().dg_approx(x) + self.ddf.dot(x-self.x)

    def ddg_approx(self, x):
        return super().ddg_approx(x) + self.ddf
