from problems.n_dim.square import Square
from sao.mappings.intervening import Exponential as Exp
import numpy as np
import pytest
from sao.mappings.mapping import MixedMapping as MM


def test_mm():
    prob = Square(4)
    f = prob.g(prob.x0)
    df = prob.dg(prob.x0)
    ddf = prob.ddg(prob.x0)

    mymap = MM(prob.n, prob.m + 1)
    mymap.add_map(Exp(p=-1), 0, 1)
    mymap.update(prob.x0, df, ddf)


if __name__ == "__main__":
    test_mm()
