from interior_point import InteriorPoint
import numpy as np
from copy import deepcopy

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

    def __init__(self, problem, x, **kwargs):
        super().__init__(problem, **kwargs)

        self.c = kwargs.get('c', 1e3)

        self.w = [x,
                  np.ones(self.m),
                  np.max(1/(x - self.alpha), 1),
                  np.max(1/(self.alpha - x), 1),
                  np.ones(self.m)]

        self.r = [np.zeros(self.n),
                  np.zeros(self.m),
                  np.zeros(self.n),
                  np.zeros(self.n),
                  np.zeros(self.m)]

        self.dw = deepcopy(self.r)
        self.wold = deepcopy(self.w)

    def get_step_size(self):
        pass

    def get_residual(self):
        pass

    def get_newton_direction(self):
        pass



