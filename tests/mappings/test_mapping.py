import numpy as np
import pytest
from sao.mappings.mapping import Mapping, Linear


class Dummy(Mapping):
    g0 = 0
    dg0 = 1
    ddg0 = 0

    def _update(self, x0, dg0, ddg0=0):
        self.g0 += x0
        self.dg0 *= dg0
        # I don't like this line, to the users adding this line does not add anything

    def _g(self, x): return self.g0 + x

    def _dg(self, x): return self.dg0 * x

    def _ddg(self, x): return self.ddg0 + x


class TestMapping:
    def test_init(self):
        """
        Test empty initialization sets child to Linear
        """

        mymap = Dummy(Dummy())

        assert isinstance(mymap, Dummy)
        assert isinstance(mymap.child, Dummy)
        assert isinstance(mymap.child.child, Linear)

    @pytest.mark.parametrize('x', [-1, 0, 1, 2, 10, 1000])
    def test_g_dg_ddg(self, x):
        """
        Test the responses of a chain of Mappings using Dummy.
        """

        mymap2 = Dummy(Dummy())

        # f[x] = f[g[x]]
        assert mymap2.g(x) == x

        # f'[x] = f'[g[x]]*g'[x]
        assert mymap2.dg(x) == x ** 2

        # f''[x] = f''[g[x]]*(g'[x])^2 + f'[g[x]]*g''[x]
        assert mymap2.ddg(x) == x ** 3 + x ** 2

    @pytest.mark.parametrize('x', [-1, 0, 1, 2, 10, 1000])
    def test_update(self, x):
        mymap = Dummy(Dummy())
        mymap.update(0, 1)

        assert mymap.g(x) == x
        assert mymap.dg(x) == x ** 2
        assert mymap.ddg(x) == x ** 3 + x ** 2
