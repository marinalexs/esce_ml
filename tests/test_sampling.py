from unittest import TestCase
from esce.data import get_mnist

class TestSampling(TestCase):
    def test_example(self):
        # Dummy code for now
        x,y = get_mnist()
        self.assertTrue(x is not None)