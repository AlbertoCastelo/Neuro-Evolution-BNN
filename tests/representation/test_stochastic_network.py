from unittest import TestCase, skip

import torch
from neat.representation.stochastic_network import StochasticNetwork
from tests.config_files.config_files import create_configuration
from tests.utils.generate_genome import generate_genome_with_hidden_units


class TestStochasticFeedForwardWithoutHiddenLayers(TestCase):

    @skip('WIP')
    def test_network_structure(self):
        self.config = create_configuration(filename='/miso.json')
        genome = generate_genome_with_hidden_units(n_input=self.config.n_input,
                                                   n_output=self.config.n_output)

        model = StochasticNetwork(genome=genome)

        input = torch.tensor([1.0, -1.0])
        result = model.forward(input.data)

        self.assertEqual(len(result), self.config.n_output)
