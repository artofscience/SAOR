from dataclasses import dataclass
from .problems import Problem, Subproblem


@dataclass
class Storage:
    problem: Problem
    subproblem: Subproblem
    # TODO: add solver as well
