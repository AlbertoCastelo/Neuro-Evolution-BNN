from unittest import TestCase

from neat.dataset.classification_mnist import MNISTDataset
from neat.dataset.classification_mnist_binary import MNISTBinaryDataset


class TestClassificationMNIST(TestCase):
    def test_get_data(self):
        dataset = MNISTBinaryDataset()
        dataset.generate_data()

        self.assertSetEqual({1, 2}, set(dataset.y.numpy()))

