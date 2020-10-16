from unittest import TestCase
from crit4nonlin.data import get_mnist

class TestSampling(TestCase):
    def test_example(self):
        # Dummy code for now
        x,y = get_mnist()
        self.assertTrue(x is not None)