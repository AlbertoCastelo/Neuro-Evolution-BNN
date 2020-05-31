from unittest import TestCase
import torch

from neat.dataset.classification_titanic import TitanicDataset, Titanic2Dataset


class TestTitanicDataset(TestCase):

    def test_generate_data(self):
        dataset = Titanic2Dataset(train_percentage=0.75)
        dataset.generate_data()

        x = dataset.x
        y = dataset.y
        print(len(dataset))
        print(y.sum())

        self.assertEqual(type(x), torch.Tensor)
        self.assertEqual(type(y), torch.Tensor)
        self.assertEqual(len(x), len(y))
        self.assertEqual(y.shape, (len(y),))
