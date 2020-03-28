from unittest import TestCase

from config_files.configuration_utils import create_configuration
from neat.dataset.classification_mnist import MNISTDataset


class TestClassificationMNIST(TestCase):
    def test_get_data(self):
        DATASET = 'mnist_downsampled'

        config = create_configuration(filename=f'/{DATASET}.json')
        config.n_output = 2

        dataset = MNISTDataset()

        x, y = dataset[0]
        print(x)
        print(y)
