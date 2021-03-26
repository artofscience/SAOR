from .interior_point import InteriorPoint
import numpy as np
from copy import deepcopy
from scipy.sparse import diags

"""
This is an addition to pdip, which extends problem P

Problem P:

min     g0[x]                       (objective)
s.t.    gi[x] <= ri,    i = 1...m   (constraints)
        aj <= xj <= bj  i = 1...n   (bound constraints)
        
and includes artificial variables y (m x 1) and z (1).


Consider the enriched Part

Problem Part:

min     g0[x] + a0 * z + sum(ci * yi + 0.5 * di * yi^2) (objective)
s.t.    gi[x] - ai * z - yi <= ri,    i = 1...m         (constraints)
        aj <= xj <= bj  i = 1...n                       (bound constraints)
        yi >= 0                                         (bound constraints)
        z >= 0
        
Constant real numbers: 
a0 > 0
ai >= 0
ci >= 0
di >= 0
ci + di > 0
ai * ci > a0 for all i with ai > 0

This makes it easier for the user to formulate certain subclasses of problems,
but also ensures a global optimum for problems that are infeasible using only the natural optimization variables.
However, in that case yi > 0.

To make Part (almost) equal to P, let a0 = 1 and ai = 0 for all i > 0.
Then z = 0 in any optimal solution of Part.
Further, let di = 1 and ci = "large", so that variables yi become relatively expensive.
Typically, y = 0 in any optimal solution of Part, and the corresponding x is an optimal solution of P as well.

The user should avoid "extremely large" value for ci (e.g. 10^10). 
It is good practice to start with low values for ci (e.g. 10^3) 
and raise if not all yi* = 0. (star denotes optimal solution)


Lagrangian L:

L := g0[x] + a0*z + sum(ci*yi + 0.5*di*yi^2) + 
    sum(lami * (gi[x] - ai*z - yi - ri)) + 
    sum(xsij*(aj - xj) + etaj*(xj - bj)) -
    sum(muj*yj) - zeta*z
    
lami    >= 0    := Lagrange multipliers wrt     gi[x] - ai * z - yi <= ri
xsij    >= 0    := Lagrange multipliers wrt     aj <= xj
etaj    >= 0    := Lagrange multipliers wrt     xj <= bj
muj     >= 0    := Lagrange multipliers wrt     yi >= 0
zeta    >= 0    := Lagrange multiplier wrt      z >= 0

L           = psi[x,lam] + 
                sum(ci*yi + 0.5*di*yi^2 - lami*(ai*z + yi + ri) - mui*yi) + 
                sum(alphaj * (aj - xj) + betaj * (xj - bj)) + 
                (a0 - zeta)*z
psi[x,lam]  = g0[x] + sum(lami * gi[x])

KKT conditions:

psi/dxj - xsij + etaj   =   0       (dL/dxj = 0)
ci + di*yi - lami - mui =   0       (dL/dyi = 0)
a0 - zeta - lami*ai     =   0       (dL/dz = 0)

gi[x] - ai*z - yi - ri  <=  0       (primal feasibility)
aj - xj                 <=  0       (primal feasibility)
xj - bj                 <=  0       (primal feasibility)

etaj                    >=  0       (dual feasibility)
xsij                    >=  0       (dual feasibility)
lami                    >=  0       (dual feasibility)
zeta                    >=  0       (dual feasibility)
mui                     >=  0       (dual feasibility)

lami * (gi[x] - ai*z - yi - ri)     =   0       (complementary slackness)
xsij * (aj - xj)                    =   0       (complementary slackness)
etaj * (xj - bj)                    =   0       (complementary slackness)
-mui*yi                             =   0       (complementary slackness)
-zeta*z                             =   0       (complementary slackness)

The zeros in the right hand sides of the complementary slackness conditions are replaced by
a "small" negative parameter epsi > 0.

Slack variables si are introduced for the constraints.

RELAXED KKT conditions:

dL/dxj, dL/dyi and dL/dz are zero

gi[x] - ai*z - yi - ri +si  <=  0       (primal feasibility)
aj - xj                     <   0       (primal feasibility)
xj - bj                     <   0       (primal feasibility)

etaj                    >  0       (dual feasibility)
xsij                    >  0       (dual feasibility)
lami                    >  0       (dual feasibility)
zeta                    >  0       (dual feasibility)
mui                     >  0       (dual feasibility)
z                       >  0
yi                      >  0
si                      >  0

lami * (gi[x] - ai*z - yi - ri) - epsi     =   0       (complementary slackness)
xsij * (aj - xj) - epsi                    =   0       (complementary slackness)
etaj * (xj - bj) - epsi                    =   0       (complementary slackness)
-mui*yi - epsi                             =   0       (complementary slackness)
-zeta*z - epsi                             =   0       (complementary slackness)
lami*si - epsi                             =   0

Given a point w = (x,xsi,eta,lam,s,y,mu,z,zeta) which satisfies feasibility
one can apply Newton's method to obtain dw = (dx,dy,dz,dlam,dxsi,deta,dmu,dzeta,ds).
Here dxsi, deta and ds can be eliminated without severe computational effort.

Subsequently we are left with a reduced system in terms of dx and dlam
"""


class InteriorPointArtificial(InteriorPoint):
    def __init__(self, problem, **kwargs):
        super().__init__(problem, **kwargs)

        self.c = kwargs.get('c', 1000*np.ones(self.m))
        self.a0 = kwargs.get('a0', 1)
        self.d = kwargs.get('d', np.zeros(self.m))
        self.a = kwargs.get('a', np.zeros(self.m))

        """
        Svanberg's implementation uses w.x = (a + b)/2.
        I found w.x = x (that is use the old variable field as initial guess)
        to converge much faster.
        Note however that a < x < b must hold. For variables where this does not hold one should use
        w.x = (a + b)/2
        """
        #FIXME: implement correct initialization
        self.w = [0.5*(self.alpha + self.beta),  # x
                  np.maximum(1/(self.x - self.alpha), 1),  # xsi
                  np.maximum(1/(self.beta - self.x), 1),  # eta
                  np.ones(self.m),  # lam
                  np.ones(self.m),  # s
                  np.ones(self.m),  # y
                  np.maximum(self.c/2, 1),  # mu
                  np.ones(1),  # z
                  np.ones(1)]  # zeta

        self.r = [np.zeros(self.n),
                  np.zeros(self.n),
                  np.zeros(self.n),
                  np.zeros(self.m),
                  np.zeros(self.m),
                  np.zeros(self.m),
                  np.zeros(self.m),
                  np.zeros(1),
                  np.zeros(1)]

        self.dw = [np.zeros(self.n),
                  np.zeros(self.n),
                  np.zeros(self.n),
                  np.zeros(self.m),
                  np.zeros(self.m),
                  np.zeros(self.m),
                  np.zeros(self.m),
                  np.zeros(1),
                  np.zeros(1)]

        self.wold = [np.zeros(self.n),
                  np.zeros(self.n),
                  np.zeros(self.n),
                  np.zeros(self.m),
                  np.zeros(self.m),
                  np.zeros(self.m),
                  np.zeros(self.m),
                  np.zeros(1),
                  np.zeros(1)]



    def get_residual(self):
        """
        r(x)        = psi / dx - xsi + eta = dg/dx[x] obj + lam' * dg/dx[x] constraints - xsi + eta
        r(xsi)      = xsi * (x - a) - e
        r(eta)      = eta * (b - x) - e
        r(lam)      = g[x] + s - z*a - y
        r(s)        = lam * si - e

        r(y)        = c + d*y - lam - mu
        r(mu)       = mu*y - e
        r(z)        = a0 - zeta - lam.a
        r(zeta)     = z*zeta - e
        """

        self.r[0][:] = self.dg(self.w[0])[0] + self.w[3].dot(self.dg(self.w[0])[1:]) - self.w[1] + self.w[2]  #rex
        self.r[1][:] = self.w[1] * (self.w[0] - self.alpha) - self.epsi  #rexsi
        self.r[2][:] = self.w[2] * (self.beta - self.w[0]) - self.epsi  #reeta
        self.r[3][:] = self.g(self.w[0])[1:] + self.w[4] - self.w[7]*self.a - self.w[5] #relam
        self.r[4][:] = self.w[3] * self.w[4] - self.epsi  #res
        self.r[5][:] = self.c + self.d * self.w[5] - self.w[6] - self.w[3]  #rey
        self.r[6][:] = self.w[6] * self.w[5] - self.epsi  #remu
        self.r[7][:] = self.a0 - self.w[8] - self.w[3].dot(self.a)  #rez
        self.r[8][:] = self.w[7] * self.w[8] - self.epsi  #rezet

    def get_newton_direction(self):
        # Some calculations to omit repetitive calculations later on
        a = self.w[0] - self.alpha
        b = self.beta - self.w[0]
        g = self.g(self.w[0])
        dg = self.dg(self.w[0])
        ddg = self.ddg(self.w[0])

        # delta_lambda
        delta_lambda = g[1:] - self.a*self.w[7] - self.w[5] + self.epsi / self.w[3]
        delta_x = dg[0] + self.w[3].dot(dg[1:]) - self.epsi/a + self.epsi/b
        delta_y = self.c + self.d*self.w[5] - self.w[3] - self.epsi/self.w[5]
        delta_z = self.a0 - self.w[3].dot(self.a) - self.epsi/self.w[7]

        diag_lambda = self.w[4]/self.w[3]  # s./lam
        diag_x = ddg[0] + self.w[3].dot(ddg[1:]) + self.w[1]/a + self.w[2]/b
        diag_y = self.d + self.w[6]/self.w[5]

        diag_lambday = diag_lambda + 1/diag_y
        delta_lambday = delta_lambda + delta_y/diag_y

        #FIXME: the m > n needs to be checked
        if self.m > self.n:
            # dldl = delta_lambda/diag_lambda
            Bx = -delta_x - (delta_lambday/diag_lambday).dot(dg[1:])
            Ax = diags(diag_x) + dg[1:].transpose().dot(diags(1/diag_lambday) * dg[1:])
            ax = - (self.a/diag_lambday).dot(dg[1:])
            az = self.w[8]/self.w[7] + - (self.a/diag_lambday).dot(self.a)

            # solve for dx
            X = np.linalg.solve(np.block([[Ax, ax], [ax.transpose(), az]]),
                                            np.block([[Bx], [-delta_z + - (delta_lambday/diag_lambday).dot(self.a)]]))  # n x n
            self.dw[0][:] = X[:-1]
            self.dw[7][:] = X[-1]
            self.dw[3][:] = (delta_lambday + self.dw[0].dot(dg[1:]) + self.dw[7]*self.a)/diag_lambday

        else:
            dxdx = delta_x/diag_x
            Blam = delta_lambday - dxdx.dot(dg[1:].transpose())
            Alam = diags(diag_lambday) + dg[1:].dot(diags(1/diag_x) * dg[1:].transpose())  # calculate dx[lam]

            # solve for dlam
            X = np.linalg.solve(np.block([[Alam, self.a], [self.a.transpose(), -self.w[8]/self.w[7]]]), np.block([[Blam], [delta_z]]))
            self.dw[3][:] = X[:-1]  # m x m
            self.dw[7][:] = X[-1]
            self.dw[0][:] = -dxdx - (self.dw[3].dot(dg[1:]))/diag_x

        # get dxsi[dx], deta[dx] and ds[dlam]
        self.dw[1][:] = -self.w[1] + self.epsi/a - (self.w[1] * self.dw[0])/a
        self.dw[2][:] = -self.w[2] + self.epsi/b + (self.w[2] * self.dw[0])/b
        self.dw[4][:] = -self.w[4] + self.epsi/self.w[3] - (self.w[4] * self.dw[3])/self.w[3]
        self.dw[5][:] = (self.dw[3] - delta_y)/diag_y
        self.dw[6][:] = -self.w[6] + self.epsi/self.w[5] - (self.w[6] * self.dw[5])/self.w[5]
        self.dw[8][:] = -self.w[8] + self.epsi/self.w[7] - (self.w[8] * self.dw[7])/self.w[7]
        # print(self.dw[1])
        # print("y:", self.w[5], "z:", self.w[7])

