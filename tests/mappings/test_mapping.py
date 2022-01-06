import pytest
import numpy as np
from sao.mappings.taylor import Taylor1
from sao.mappings.exponential import Exponential
from Problems.svanberg1987 import CantileverBeam


def test_mapping():
    prob = CantileverBeam()

    x = prob.x0

    f = prob.g(x)
    df = prob.dg(x)
    ddf = prob.ddg(x)

    map1 = Exponential(-1)
    map2 = Exponential(map1)
    map3 = Taylor1(map2)
    map3.update(x,f,df,ddf)


    print(map1.g(x))
    print(map2.g(x))
    print(map3.g(x))
    print(map1.g(10))
    print(map2.g(10))
    print(map3.g(10))






if __name__ == "__main__":
    test_mapping()