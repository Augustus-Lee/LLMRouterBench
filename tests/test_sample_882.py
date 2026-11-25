# -*- coding: utf-8 -*-
"""sample_882 抽样采样函数测试。"""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.sample_882 import DATASET_SPECS, sample_indices  # noqa: E402


class TestSampleIndices(unittest.TestCase):
    def test_deterministic(self):
        self.assertEqual(sample_indices(100, 20, 42), sample_indices(100, 20, 42))

    def test_size_and_unique(self):
        idxs = sample_indices(100, 20, 42)
        self.assertEqual(len(idxs), 20)
        self.assertEqual(len(set(idxs)), 20)
        self.assertTrue(all(0 <= i < 100 for i in idxs))

    def test_sorted(self):
        self.assertEqual(sample_indices(100, 20, 42), sorted(sample_indices(100, 20, 42)))

    def test_sample_larger_than_population(self):
        self.assertEqual(sample_indices(5, 100, 42), [0, 1, 2, 3, 4])

    def test_full_population(self):
        self.assertEqual(set(sample_indices(5, 5, 42)), {0, 1, 2, 3, 4})

    def test_zero_cases(self):
        self.assertEqual(sample_indices(0, 10, 42), [])
        self.assertEqual(sample_indices(10, 0, 42), [])

    def test_different_seed_differs(self):
        self.assertNotEqual(sample_indices(500, 50, 1), sample_indices(500, 50, 2))


class TestDatasetSpecs(unittest.TestCase):
    def test_quota_sum_882(self):
        total = sum(spec[1] for spec in DATASET_SPECS.values())
        self.assertEqual(total, 882)

    def test_quota_per_dataset(self):
        self.assertEqual(DATASET_SPECS["gpqa"][1], 150)
        self.assertEqual(DATASET_SPECS["mmlupro"][1], 150)
        self.assertEqual(DATASET_SPECS["math500"][1], 150)
        self.assertEqual(DATASET_SPECS["arcc"][1], 150)
        self.assertEqual(DATASET_SPECS["gsm8k"][1], 100)
        self.assertEqual(DATASET_SPECS["mmlu"][1], 182)


if __name__ == "__main__":
    unittest.main()
