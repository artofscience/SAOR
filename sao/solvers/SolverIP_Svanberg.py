## Imports
import numpy as np


## Svanberg's InteriorPoint solver found in http://www.ingveh.ulg.ac.be/uploads/education/meca-0027-1/MMA_DCAMM_1998.pdf
class SvanbergIP:

    ## Constructor of class
    def __init__(self, n, m):
        self.n = n
        self.m = m
        self.epsimin = 1e-6
        self.maxittt = 20
        self.solves = 0
        self.counter = 0
        self.move_lim = 0.15
        self.cCoef = 1e3
        self.name = 'SvanbergIP'

    ## Subsolv function
    def subsolv(self, subprob, **kwargs):
        """
        This function subsolv solves the approximate subproblem P_NLP_tilde:

        minimize              SUM[ g_0_tilde ] + a0*z + + SUM[ ci*yi + 0.5*di*(yi)^2 ]

        subject to            SUM[ g_i_tilde ] - ai*z - yi <= bi,
                              alpha_j <=  xj <=  beta_j,
                              yi >= 0,
                              z >= 0.
        Input:  subprob
        Output: x, y, z, lam, xsi, eta, mu, zet, s
        """


        # Initialization of parameters (once per design iteration)
        a0 = 1.0
        a = np.zeros(self.m)
        c = self.cCoef * np.ones(self.m)
        d = np.zeros(self.m)  # self.d = np.ones((self.m + 1, 1))
        epsi = 1
        x = 0.5 * (subprob.alpha + subprob.beta)
        y = np.ones(self.m)
        z = 1
        lam = np.ones(self.m)
        xsi = np.maximum((1.0 / (x - subprob.alpha)), 1)
        eta = np.maximum((1.0 / (subprob.beta - x)), 1)
        mu = np.maximum(1, 0.5 * c)
        zet = 1
        s = np.ones(self.m)
        itera = 0
        ittt = 0

        self.counter = 0

        while epsi > self.epsimin:
            self.counter += 1

            # upcoming lines determine the left hand sides, i.e. the resiudals of all constraints
            residunorm, residu = self.residual(x, y, z, lam, xsi, eta, mu, zet, s, epsi, a0, a, c, d, subprob)
            residumax = np.max(np.abs(residu))

            # Newton step (Section 7.3)
            # the algorithm is terminated when the maximum residual has become smaller than 0.9*epsi
            # and epsi has become sufficiently small
            ittt = 0
            while (residumax > 0.9 * epsi) and (ittt < self.maxittt):
                ittt = ittt + 1
                itera = itera + 1

                # Calculating PSIjj
                g_j_tilde_value = subprob.g(x)
                dg_j_tilde_value = subprob.dg(x)
                dg_j_tilde2_value = subprob.ddg(x)
                dpsi_dx = (dg_j_tilde_value[0, :] + np.dot(lam, dg_j_tilde_value[1:, :]))
                d2psi_dx2 = (dg_j_tilde2_value[0, :] + np.dot(lam, dg_j_tilde2_value[1:, :]))

                # Calculation of right hand sides of partially reduced system (Svanberg1998/page 16)
                delx = dpsi_dx - epsi / (x - subprob.alpha) + epsi / (subprob.beta - x)
                dely = c + d * y - lam - epsi / y
                delz = a0 - np.dot(a, lam) - epsi / z
                dellam = g_j_tilde_value[1:] - a * z - y + epsi / lam

                # Calculation of diagonal matrices: Dx, Dy, Dlam
                diagx = d2psi_dx2 + xsi/(x - subprob.alpha) + eta/(subprob.beta - x)
                diagy = d + mu / y
                diaglam = s / lam                           # - is missing
                diaglamyi = diaglam + 1.0 / diagy           # what is that?

                # form reduced system of Eq(7.9)
                blam = dellam + dely / diagy - np.dot(dg_j_tilde_value[1:, :], (delx / diagx))
                bb = np.hstack((blam, delz))
                Alam = np.diag(diaglamyi) + np.einsum('ij,j,kj->ik', dg_j_tilde_value[1:, :], (1 / diagx),
                                                      dg_j_tilde_value[1:, :])
                AA = np.hstack((np.vstack((Alam, a.T)), np.array([np.hstack((a, -zet / z))]).T))

                # solve system for delta lambda and delta z
                solut = np.linalg.solve(AA, bb)

                # solution of delta vars
                dlam = solut[0:self.m]
                dz = solut[self.m]
                dx = -delx / diagx - np.dot(dg_j_tilde_value[1:, :].T, dlam) / diagx
                dy = -dely / diagy + dlam / diagy
                dxsi = -xsi + epsi / (x - subprob.alpha) - (xsi * dx) / (x - subprob.alpha)
                deta = -eta + epsi / (subprob.beta - x) + (eta * dx) / (subprob.beta - x)
                dmu = -mu + epsi / y - (mu * dy) / y
                dzet = -zet + epsi / z - zet * dz / z
                ds = -s + epsi / lam - (s * dlam) / lam

                # Form inequalities of page 17 (theta = steg). Inequalities become equalities cuz we want max{theta}
                xx = np.hstack((y, z, lam, xsi, eta, mu, zet, s))
                dxx = np.hstack((dy, dz, dlam, dxsi, deta, dmu, dzet, ds))

                # calculate the step-size
                stepxx = -1.01 * dxx / xx
                stmxx = np.max(stepxx)

                stepalpha = -1.01 * dx / (x - subprob.alpha)
                stmalpha = np.max(stepalpha)

                stepbeta = 1.01 * dx / (subprob.beta - x)
                stmbeta = np.max(stepbeta)

                # Step-size calculation: We 're looking for the max{theta} that satisfies the above equalities
                steg = 1.0 / np.maximum.reduce([stmalpha, stmbeta, stmxx, 1])

                # set old variables
                xold = x.copy()
                yold = y.copy()
                zold = z
                lamold = lam.copy()
                xsiold = xsi.copy()
                etaold = eta.copy()
                muold = mu.copy()
                zetold = zet
                sold = s.copy()

                # why put net residual twice the initial norm?
                resinew = 2 * residunorm

                # Line search to find the optimal step-size
                itto = 0
                while (resinew > residunorm) and (itto < self.maxittt):
                    itto = itto + 1

                    # Calculate new point in the line-search
                    x = xold + steg * dx
                    y = yold + steg * dy
                    z = zold + steg * dz
                    lam = lamold + steg * dlam
                    xsi = xsiold + steg * dxsi
                    eta = etaold + steg * deta
                    mu = muold + steg * dmu
                    zet = zetold + steg * dzet
                    s = sold + steg * ds

                    # Recalculate residual of equations
                    resinew, residu = self.residual(x, y, z, lam, xsi, eta, mu, zet, s, epsi, a0, a, c, d, subprob)

                    # Reduce step-size by 50%
                    steg = steg / 2

                residunorm = resinew
                residumax = np.max(np.abs(residu))

                # Double the step-size
                steg *= 2

            # Decrease epsilon with factor 10
            epsi *= 0.1

            # print("iter: %2d" % self.counter, ", ",
            #       "solves: %2d" % ittt, ", ",
            #       "obj: %.2e" % subprob.g(x)[0], ", ",
            #       "x:", ["{:+.2f}".format(i) for i in x[1:3]], ", ",
            #       "lam:", ["{:+.2f}".format(i) for i in lam])

        return x, y, z, lam, xsi, eta, mu, zet, s

    ## Calculates the residual of the relaxed KKT conditions
    @staticmethod
    def residual(x, y, z, lam, xsi, eta, mu, zet, s, epsi, a0, a, c, d, subprob):

        # Calculating g_j_tilde_value, dg_j_tilde_value and dpsi_dx
        g_j_tilde_value = subprob.g(x)
        dg_j_tilde_value = subprob.dg(x)
        dpsi_dx = (dg_j_tilde_value[0, :] + np.dot(lam.T, dg_j_tilde_value[1:, :]))

        # Calculation of other residuals
        rex = dpsi_dx - xsi + eta
        rey = c + d * y - mu - lam
        rez = a0 - zet - np.dot(a, lam)
        relam = g_j_tilde_value[1:] - a * z - y + s
        rexsi = xsi * (x - subprob.alpha) - epsi
        reeta = eta * (subprob.beta - x) - epsi
        remu = mu * y - epsi
        rezet = zet * z - epsi
        res = lam * s - epsi

        # Put all residuals in one line
        residu = np.hstack((rex, rey, rez, relam, rexsi, reeta, remu, rezet, res))

        # Euclidean norm calculation
        resinew = np.sqrt(np.dot(residu.T, residu))
        return resinew, residu
