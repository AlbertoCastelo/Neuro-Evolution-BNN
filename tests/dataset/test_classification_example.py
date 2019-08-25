from unittest import TestCase
import numpy as np
import torch

from neat.dataset.classification_example import ClassificationExample1Dataset


class TestClassificationExample1Dataset(TestCase):
    def setUp(self) -> None:
        pass

    def test__get_x_y(self):
        dataset = ClassificationExample1Dataset()
        x = np.zeros((3, 2))
        x[1][1] = 6.0
        x[2][0] = 2.0
        x, y = dataset._get_x_y(x)

        self.assertEqual(type(x), np.ndarray)
        self.assertEqual(x.shape, (3, 2))
        self.assertEqual(type(y), np.ndarray)

    def test_generate_data(self):
        dataset = ClassificationExample1Dataset()
        dataset.generate_data()

        x = dataset.x
        y = dataset.y

        self.assertEqual(type(x), torch.Tensor)
        self.assertEqual(type(y), torch.Tensor)
        self.assertEqual(len(x), len(y))

