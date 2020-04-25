from unittest import TestCase

from config_files.configuration_utils import create_configuration
from neat.dataset.classification_mnist_binary import MNISTBinaryDataset


class TestClassificationMNIST(TestCase):
    def setUp(self) -> None:
        self.config = create_configuration(filename='/mnist_binary.json')
        self.config.n_output = 2

    def test_get_data(self):
        dataset = MNISTBinaryDataset(train_percentage=0.5)
        dataset.generate_data()

        self.assertSetEqual({0, 1}, set(dataset.y.numpy()))

