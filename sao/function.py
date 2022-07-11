import numpy as np
import abc
#
class Function(abc.ABC):
#
    def __init__(self, name='function', n=0):
        if n == 0: print('Error')
        self.name = name
        self.n = n
        self.x_k = np.zeros(n,dtype=float)
        self.g_k = 0e0
        self.dg_k = np.zeros(n,dtype=float)
        self.y_k = np.zeros(n,dtype=float)
        self.dy_k = np.zeros(n,dtype=float)
        self.k = -1
        self.hst_x_k  = []
#
    def __str__(self):
        return f"Function {self.name} of dimension {self.n}"
#
    def setpoint(self, x, f, df, aux, k_s=1, store=False):
#
        self.k = self.k + 1
#
        if isinstance(f,float): self.g_k = f
        else: print('Error 1')
        for i in range(self.n):
            if isinstance(x[i],float): self.x_k[i] = x[i]
            else: print('Error 2')
            if isinstance(df[i],float): self.dg_k[i] = df[i]
            else: print('Error 3')
#
        if store:
            self.hst_x_k.append(self.x_k.copy())
            if len(self.hst_x_k) > k_s:
                self.hst_x_k.pop(0)
#
        self.parameters(aux)
        self.y_k, self.dy_k, _, _ = self.intercurve(x)
#
    def domain(self):
        d_l = -1e8*np.ones(self.n,dtype=float)
        d_u = 1e8*np.ones(self.n,dtype=float)
        return d_l, d_u
#
    def g(self, x):
        g = self.g_k
        x_k = self.x_k
        dg_k = self.dg_k
        y_k = self.y_k
        dy_k = self.dy_k
        y, _, _, c_x = self.intercurve(x)
        g = g + np.dot(dg_k/dy_k,(y - y_k)) + np.dot(c_x/2e0,(x-x_k)**2e0)
        return g
    def dg(self, x):
        g = self.g_k
        x_k = self.x_k
        dg_k = self.dg_k
        y_k = self.y_k
        dy_k = self.dy_k
        y, dy, _, c_x = self.intercurve(x)
        dg = np.zeros_like(self.dg_k)
        dg[:] = dg_k/dy_k*dy + c_x*(x-x_k)
        return dg
    def ddg(self, x):
        g = self.g_k
        x_k = self.x_k
        dg_k = self.dg_k
        y_k = self.y_k
        dy_k = self.dy_k
        _, _, ddy, c_x = self.intercurve(x)
        ddg = np.zeros_like(self.dg_k)
        ddg[:] = dg_k/dy_k*ddy + c_x
        return ddg
#
#   def evaluate(self, x):
#
#       g = self.g_k
#       dg = np.zeros_like(self.dg_k)
#       ddg = np.zeros_like(self.dg_k)
#
#       x_k = self.x_k
#       dg_k = self.dg_k
#       y_k = self.y_k
#       dy_k = self.dy_k
#
#       y, dy, ddy, c_x = self.intercurve(x)
#
#       for i in range(self.n):
#           g = g + dg_k[0][i]/dy_k[i]*(y[i] - y_k[i]) + c_x[i]/2e0*(x[i]-x_k[i])**2e0
#           dg[0][i] = dg_k[0][i]/dy_k[i]*dy[i] + c_x[i]*(x[i]-x_k[i])
#           ddg[0][i] = dg_k[0][i]/dy_k[i]*ddy[i] + c_x[i]
#
#       return g, dg, ddg
#
    def parameters(self,prob):
        pass
#
    @abc.abstractmethod
    def intercurve(self,x):
        y = np.ones_like(x)
        dy = np.ones_like(x)
        ddy = np.zeros_like(x)
        c_x = np.zeros_like(x)
        return y, dy, ddy, c_x
#
