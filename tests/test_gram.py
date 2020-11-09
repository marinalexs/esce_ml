from unittest import TestCase
from esce.models import get_gram_triu, get_gram, probe_gram_triu, KernelType
import numpy as np

class TestGram(TestCase):
    def test_gram_triu(self):
        num_samples = 100
        X = np.random.random((num_samples, 100))
        triu = get_gram_triu(X, KernelType.RBF, gamma=0.5)
        num_elements = num_samples * (num_samples+1) / 2
        self.assertEqual(len(triu), num_elements)

    def test_gram(self):
        num_samples = 100
        X = np.random.random((num_samples, 100))
        gram = get_gram(X, KernelType.RBF, gamma=0.5)
        self.assertEqual(gram.shape, (num_samples, num_samples))

    def test_gram_cache(self):
        num_samples = 100
        X = np.random.random((num_samples, 100)).astype("f")
        get_gram(X, KernelType.SIGMOID, gamma=0.8)
        self.assertTrue(probe_gram_triu(X, KernelType.SIGMOID, gamma=0.8))