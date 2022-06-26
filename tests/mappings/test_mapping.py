import numpy as np
import pytest
from sao.mappings.mapping import Mapping, Linear


class Dummy(Mapping):
    def _g(self, x): return x

    def _dg(self, x): return x

    def _ddg(self, x): return x


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
    def test_update(self, x):
        """
        Test the update function of a chain of Mappings using Dummy.
        """

        mymap2 = Dummy(Dummy())

        # f[x] = f[g[x]]
        assert mymap2.g(x) == x

        # f'[x] = f'[g[x]]*g'[x]
        assert mymap2.dg(x) == x ** 2

        # f''[x] = f''[g[x]]*(g'[x])^2 + f'[g[x]]*g''[x]
        assert mymap2.ddg(x) == x ** 3 + x ** 2
