from funcarray import array


class TestElementSum():
    def test_zero_sum(self):
        a = array((10, 10), lambda x, y: 0)
        assert a.sum() == 0.0
