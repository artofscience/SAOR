from sao.approximations.mma import MMA
from sao.problems.rosenbrock import Rosenbrock


def test_mma():

    problem = Rosenbrock()

    # FIXME: it would be nicer to support passing `problem` as a whole, see

    # approximation = MMA(problem.n, problem.m, problem.xmin, problem.xmax)
    approximation = MMA(problem)

    assert approximation.n == problem.n
