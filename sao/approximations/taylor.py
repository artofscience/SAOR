from .approximation import Approximation


class Taylor1(Approximation):
    def g(self, x):
        return self.f + self.df.dot(x-self.x)

    def dg(self, x):
        return self.df

    def ddg(self, x):
        return 1.0


class Taylor2(Taylor1):
    def g(self, x):
        return super().g(x) + 1/2*self.ddf.dot((x-self.x)**2)

    def dg(self, x):
        return super().dg(x) + self.ddf.dot(x-self.x)

    def ddg(self, x):
        return super().ddg(x) + self.ddf
