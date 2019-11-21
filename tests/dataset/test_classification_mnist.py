from unittest import TestCase

from neat.dataset.classification_mnist import MNISTDataset


class TestClassificationMNIST(TestCase):
    def test_get_data(self):
        dataset = MNISTDataset()

        x, y = dataset[0]
        print(x)
        print(y)
