import unittest
import numpy as np
from si.statistics.tanimoto_similarity import tanimoto_similarity

class TestTanimotoSimilarity(unittest.TestCase):

    def test_identical_vectors(self):
        x = np.array([1, 0, 1, 1])
        y = np.array([[1, 0, 1, 1]])
        result = tanimoto_similarity(x, y)
        self.assertAlmostEqual(result[0], 1.0)

    def test_orthogonal_vectors(self):
        x = np.array([1, 0, 1, 0])
        y = np.array([[0, 1, 0, 1]])
        result = tanimoto_similarity(x, y)
        self.assertAlmostEqual(result[0], 0.0)

    def test_partial_overlap(self):
        x = np.array([1, 1, 0, 1])
        y = np.array([[1, 0, 1, 1]])
        result = tanimoto_similarity(x, y)
        # intersection = 2, union = 4 → 2/4 = 0.5
        self.assertAlmostEqual(result[0], 0.5)

    def test_multiple_y_samples(self):
        x = np.array([1, 1, 0])
        y = np.array([[1, 1, 0],   # exact match → 1.0
                      [1, 0, 0],   # 1 common / 2 union = 0.5
                      [0, 0, 1]])  # no overlap → 0.0
        result = tanimoto_similarity(x, y)
        expected = np.array([1.0, 0.5, 0.0])
        np.testing.assert_almost_equal(result, expected)

    def test_zero_vector(self):
        x = np.array([0, 0, 0])
        y = np.array([[0, 0, 0]])
        result = tanimoto_similarity(x, y)
        # convention: similarity of zero-vectors = 0
        self.assertEqual(result[0], 0.0)

if __name__ == '__main__':
    unittest.main()