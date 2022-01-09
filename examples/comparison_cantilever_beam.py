from problems.svanberg1987 import CantileverBeam
from sao.move_limits import Bounds, MoveLimit, MoveLimitFraction, AdaptiveMoveLimit
from sao.intervening_variables.mma import MMA87A, MMA02, MMA87C
from sao.intervening_variables import Linear, Exponential
from sao.intervening_variables.mixed_intervening import MixedIntervening
from sao.solvers.primal_dual_interior_point import pdip, Pdipx
from sao.problems.subproblem import Subproblem
from sao.approximations import Taylor1, Taylor2, SphericalTaylor2, NonSphericalTaylor2
from sao.convergence_criteria import IterationCount


"""
This example compares different SAO schemes for solving the Svanberg 1987 Two Bar Truss problem.
For a "fair" comparison, we use the convergence criteria as used in the paper.
"""

"""
We start with the scheme as presented in the paper.
"""
def original():
    problem = CantileverBeam()
    bounds = Bounds(problem.x_min, problem.x_max)
    movelimit = MoveLimitFraction(fraction=2.0)
    subproblem = Subproblem(Taylor1(MMA87A(t=1/8)), limits=[bounds, movelimit])
    optimizer(problem, subproblem, IterationCount(10))


"""
Now let's see what our "default" MMA does
"""
def mma2():
    problem = CantileverBeam()
    movelimit = MoveLimit(move_limit=1, dx=60)
    bounds = Bounds(xmin=problem.x_min, xmax=problem.x_max)
    subproblem = Subproblem(Taylor1(MMA87C()), limits=[bounds, movelimit])
    optimizer(problem, subproblem, IterationCount(10))


"""
Let's check out LP + AML
"""
def lp_aml():
    problem = CantileverBeam()
    bounds = Bounds(xmin=problem.x_min, xmax=problem.x_max)
    movelimit = AdaptiveMoveLimit(move_limit=1, dx=2)
    subproblem = Subproblem(Taylor1(), limits=[bounds, movelimit])
    optimizer(problem, subproblem, IterationCount(10))

"""
What about MMA with an Adaptive Move Limit strategy?
"""
def mma_aml():
    problem = CantileverBeam()
    bounds = Bounds(xmin=problem.x_min, xmax=problem.x_max)
    movelimit = AdaptiveMoveLimit(move_limit=0.3, dx=problem.x_max - problem.x_min)
    subproblem = Subproblem(Taylor1(MMA02(problem.x_min, problem.x_max)), limits=[bounds, movelimit])
    optimizer(problem, subproblem, IterationCount(10))


"""
Let's check a mixed scheme.
"""
def mixed_lp_mma():
    problem = CantileverBeam()
    bounds = Bounds(xmin=problem.x_min, xmax=problem.x_max)
    movelimit = AdaptiveMoveLimit(move_limit=1, dx=2)
    intvar = MixedIntervening(problem.n, problem.m+1, default=Linear())
    intvar.set_intervening(MMA87A(), var=1)
    subproblem = Subproblem(Taylor1(intvar), limits=[bounds, movelimit])
    optimizer(problem, subproblem, IterationCount(10))


"""
What about second order info?
"""
def taylor2exp(p=2):
    problem = CantileverBeam()
    bounds = Bounds(xmin=problem.x_min, xmax=problem.x_max)
    subproblem = Subproblem(Taylor2(Exponential(p=p)), limits=[bounds])
    optimizer(problem, subproblem, IterationCount(10))


"""
What about approximating curvature info by satisfying the previous point?
(see "Incomplete series expansion for function approximation" by Groenwold et al. / Eq. 16)
"""
def spherical_taylor2(p=1):
    problem = CantileverBeam()
    bounds = Bounds(xmin=problem.x_min, xmax=problem.x_max)
    subproblem = Subproblem(SphericalTaylor2(Exponential(p)), limits=[bounds])
    optimizer(problem, subproblem, IterationCount(10))


"""
What about approximating curvature info by satisfying the previous point gradient?
(see "Incomplete series expansion for function approximation" by Groenwold et al. / Eq. 23)
"""
def nonspherical_taylor2(p=1):
    problem = CantileverBeam()
    bounds = Bounds(xmin=problem.x_min, xmax=problem.x_max)
    subproblem = Subproblem(NonSphericalTaylor2(Exponential(p)), limits=[bounds])
    optimizer(problem, subproblem, IterationCount(10))


def optimizer(problem, subproblem, converged):
    x = problem.x0
    while not converged:
        f = problem.g(x)
        df = problem.dg(x)
        infeasibility = max(0.0, f[1])
        print("{}: {:.3f} {:.3f}".format(converged.iteration-1, f[0], infeasibility))
        if (infeasibility < 0.001) and (f[0] < 1.001*1.340):
            print("The optimum solution found")
            break

        ddf = problem.ddg(x) if subproblem.approx.__class__.__name__ == 'Taylor2' else None
        subproblem.build(x, f, df, ddf)
        x[:] = pdip(subproblem, variables=Pdipx)[0]
    print("\n")


if __name__ == "__main__":
    original()
    mma2()
    lp_aml()
    mma_aml()
    mixed_lp_mma()
    taylor2exp(p=-0.1)
    spherical_taylor2(-0.1)
    nonspherical_taylor2(-1)
