from sao import Storage
from sao.problems import Subproblem
from Problems.Square import Square
from sao.intervening_variables import Linear, ConLin
from sao.approximations import Taylor1
from sao.solvers.interior_point import InteriorPointX
from sao.move_limits import MoveLimitIntervening


def test_storage():
    problem = Square(5)

    subproblem = Subproblem(
            intervening=ConLin(),
            approximation=Taylor1(),
            ml=MoveLimitIntervening(xmin=problem.xmin, xmax=problem.xmax))

    storage = Storage(problem, subproblem)
    assert storage.problem == problem
    assert storage.subproblem == subproblem
