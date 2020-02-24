from unittest import TestCase
import numpy as np
import torch

from neat.dataset.classification_example import ClassificationExample1Dataset
from neat.dataset.classification_titanic import TitanicDataset


class TestTitanicDataset(TestCase):

    def test_generate_data(self):
        dataset = TitanicDataset()
        dataset.generate_data()

        x = dataset.x
        y = dataset.y

        self.assertEqual(type(x), torch.Tensor)
        self.assertEqual(type(y), torch.Tensor)
        self.assertEqual(len(x), len(y))
        self.assertEqual(y.shape, (len(y),))
