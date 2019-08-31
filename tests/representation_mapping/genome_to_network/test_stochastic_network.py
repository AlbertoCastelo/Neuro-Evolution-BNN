from unittest import TestCase

import torch

from neat.representation_mapping.genome_to_network.stochastic_network import StochasticNetworkOld
from tests.config_files.config_files import create_configuration
from tests.utils.generate_genome import generate_genome_with_hidden_units


class TestStochasticFeedForwardWithoutHiddenLayers(TestCase):

    def test_network_structure_miso(self):
        self.config = create_configuration(filename='/miso.json')
        genome = generate_genome_with_hidden_units(n_input=self.config.n_input,
                                                   n_output=self.config.n_output)

        model = StochasticNetworkOld(genome=genome)

        input = torch.tensor([1.0, -1.0])
        result = model(input.data)

        self.assertEqual(len(result), self.config.n_output)

    def test_network_structure_siso(self):
        self.config = create_configuration(filename='/siso.json')
        genome = generate_genome_with_hidden_units(n_input=self.config.n_input,
                                                   n_output=self.config.n_output)

        model = StochasticNetworkOld(genome=genome)

        input = torch.tensor([1.0])
        result = model(input.data)

        self.assertEqual(len(result), self.config.n_output)

    def test_network_structure_siso_with_batch(self):
        self.config = create_configuration(filename='/siso.json')
        genome = generate_genome_with_hidden_units(n_input=self.config.n_input,
                                                   n_output=self.config.n_output)

        model = StochasticNetworkOld(genome=genome)

        input = torch.tensor([[1.0],
                             [1.0],
                             [1.0]])
        result = model(input.data)

        self.assertEqual(result.shape[1], self.config.n_output)
        self.assertEqual(result.shape[0], 3)

    def test_same_input_gives_different_result(self):
        self.config = create_configuration(filename='/siso.json')
        genome = generate_genome_with_hidden_units(n_input=self.config.n_input,
                                                   n_output=self.config.n_output)

        model = StochasticNetworkOld(genome=genome)

        input = torch.tensor([[1.0]])
        result_1 = model(input.data)
        result_2 = model(input.data)

        self.assertNotEqual(result_1.item(), result_2.item())
