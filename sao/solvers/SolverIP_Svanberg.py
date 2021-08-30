## Imports
import numpy as np


## Svanberg's InteriorPoint solver found in http://www.ingveh.ulg.ac.be/uploads/education/meca-0027-1/MMA_DCAMM_1998.pdf
def ipsolver(problem, x0=None, epsimin=1e-6, max_inner_iter=20, max_lines_iter=20, max_outer_iter=100,
            epsifac=0.9, epsired=0.1, cCoef=1000):
    """
    This function ipsolver solves the approximate problem P:

    minimize

    .. math::

        \\sum \\tilde{g_0} + a_0*z + \\sum \\left(c_i*y_i + 0.5*d_i*(y_i)^2\\right)

    subject to

    .. math::

        \\sum \\tilde{g_i} - a_i*z - y_i \\leq b_i \\\\
        \\alpha_j \\leq  x_j \\leq  \\beta_j \\\\
        y_i \\geq 0 \\\\
        z \\geq 0

    Input:  problem
    Output: x, y, z, lam, xsi, eta, mu, zet, s
    """

    # Initialization of parameters (once per design iteration)
    n, m = problem.n, problem.m
    a0 = 1.0
    a = np.zeros(m)
    c = cCoef * np.ones(m)
    d = np.zeros(m)
    epsi = 1
    if x0 is None:
        x = 0.5 * (problem.x_min + problem.x_max)
    else:
        x = x0.copy()
    y = np.ones(m)
    z = 1
    lam = np.ones(m)
    xsi = np.maximum((1.0 / (x - problem.x_min)), 1)
    eta = np.maximum((1.0 / (problem.x_max - x)), 1)
    mu = np.maximum(1, 0.5 * c)
    zet = 1
    s = np.ones(m)
    outer_iter = 0

    # Outer iterations that aim to solve the relaxed KKT conditions
    while outer_iter < max_outer_iter and epsi > epsimin:
        outer_iter += 1

        # upcoming lines determine the left hand sides, i.e. the resiudals of all constraints
        residunorm, residu = residual(x, y, z, lam, xsi, eta, mu, zet, s, epsi, a0, a, c, d, problem)
        residumax = np.max(np.abs(residu))

        # Newton step (Section 7.3)
        # the algorithm is terminated when the maximum residual has become smaller than epsifac * epsi
        # and epsi has become sufficiently small
        inner_iter = 0
        while inner_iter < max_inner_iter and residumax > epsifac * epsi:
            inner_iter = inner_iter + 1

            # Calculating PSIjj
            g_j_tilde_value = problem.g(x)
            dg_j_tilde_value = problem.dg(x)
            dg_j_tilde2_value = problem.ddg(x)
            dpsi_dx = (dg_j_tilde_value[0, :] + np.dot(lam, dg_j_tilde_value[1:, :]))
            d2psi_dx2 = (dg_j_tilde2_value[0, :] + np.dot(lam, dg_j_tilde2_value[1:, :]))

            # Calculation of right hand sides of partially reduced system (Svanberg1998/page 16)
            delx = dpsi_dx - epsi / (x - problem.x_min) + epsi / (problem.x_max - x)
            dely = c + d * y - lam - epsi / y
            delz = a0 - np.dot(a, lam) - epsi / z
            dellam = g_j_tilde_value[1:] - a * z - y + epsi / lam

            # Calculation of diagonal matrices: Dx, Dy, Dlam
            diagx = d2psi_dx2 + xsi/(x - problem.x_min) + eta/(problem.x_max - x)
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
            dlam = solut[0:m]
            dz = solut[m]
            dx = -delx / diagx - np.dot(dg_j_tilde_value[1:, :].T, dlam) / diagx
            dy = -dely / diagy + dlam / diagy
            dxsi = -xsi + epsi / (x - problem.x_min) - (xsi * dx) / (x - problem.x_min)
            deta = -eta + epsi / (problem.x_max - x) + (eta * dx) / (problem.x_max - x)
            dmu = -mu + epsi / y - (mu * dy) / y
            dzet = -zet + epsi / z - zet * dz / z
            ds = -s + epsi / lam - (s * dlam) / lam

            # Form inequalities of page 17 (theta = steg). Inequalities become equalities cuz we want max{theta}
            xx = np.hstack((y, z, lam, xsi, eta, mu, zet, s))
            dxx = np.hstack((dy, dz, dlam, dxsi, deta, dmu, dzet, ds))

            # calculate the step-size
            stepxx = -1.01 * dxx / xx
            stmxx = np.max(stepxx)

            stepalpha = -1.01 * dx / (x - problem.x_min)
            stmalpha = np.max(stepalpha)

            stepbeta = 1.01 * dx / (problem.x_max - x)
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
            lines_iter = 0
            while lines_iter < max_lines_iter and resinew > residunorm:
                lines_iter = lines_iter + 1

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
                resinew, residu = residual(x, y, z, lam, xsi, eta, mu, zet, s, epsi, a0, a, c, d, problem)

                # Reduce step-size by 50%
                steg *= 0.5

            residunorm = resinew
            residumax = np.max(np.abs(residu))

            # Double the step-size
            steg *= 2

        # Reduce epsilon with factor epsired
        epsi *= epsired

    return x


## Calculates the residual of the relaxed KKT conditions
def residual(x, y, z, lam, xsi, eta, mu, zet, s, epsi, a0, a, c, d, problem):

    # Calculating g_j_tilde_value, dg_j_tilde_value and dpsi_dx
    g_j_tilde_value = problem.g(x)
    dg_j_tilde_value = problem.dg(x)
    dpsi_dx = (dg_j_tilde_value[0, :] + np.dot(lam.T, dg_j_tilde_value[1:, :]))

    # Calculation of other residuals
    rex = dpsi_dx - xsi + eta
    rey = c + d * y - mu - lam
    rez = a0 - zet - np.dot(a, lam)
    relam = g_j_tilde_value[1:] - a * z - y + s
    rexsi = xsi * (x - problem.x_min) - epsi
    reeta = eta * (problem.x_max - x) - epsi
    remu = mu * y - epsi
    rezet = zet * z - epsi
    res = lam * s - epsi

    # Put all residuals in one line
    residu = np.hstack((rex, rexsi, reeta, relam, res, rey, remu, rez, rezet))

    # Euclidean norm calculation
    resinew = np.linalg.norm(residu)
    return resinew, residu
