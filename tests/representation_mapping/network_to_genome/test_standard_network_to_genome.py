from unittest import TestCase

import torch

from config_files.configuration_utils import create_configuration
from neat.genome import Genome
from neat.representation_mapping.network_to_genome.standard_feed_forward_to_genome import \
    get_genome_from_standard_network
from deep_learning.standard.feed_forward import FeedForward


class TestNetwork2Genome(TestCase):
    def test_mapping(self):
        config = create_configuration(filename='/classification-miso.json')
        n_neurons_per_layer = 3
        network = FeedForward(n_input=config.n_input, n_output=config.n_output,
                              n_neurons_per_layer=n_neurons_per_layer,
                              n_hidden_layers=1)

        std = 0.1
        genome = get_genome_from_standard_network(network, std=std)
        self.assertEqual(type(genome), Genome)

        parameters = network.state_dict()
        # hidden to output
        self.assertEqual(parameters['layer_0.weight'][0, 0], genome.connection_genes[(2, 0)].weight_mean)
        self.assertEqual(parameters['layer_0.weight'][1, 0], genome.connection_genes[(2, 1)].weight_mean)
        self.assertEqual(parameters['layer_0.weight'][0, 1], genome.connection_genes[(3, 0)].weight_mean)
        self.assertEqual(parameters['layer_0.weight'][1, 1], genome.connection_genes[(3, 1)].weight_mean)
        self.assertEqual(parameters['layer_0.weight'][0, 2], genome.connection_genes[(4, 0)].weight_mean)
        self.assertEqual(parameters['layer_0.weight'][1, 2], genome.connection_genes[(4, 1)].weight_mean)

        self.assertEqual(parameters['layer_0.bias'][0], genome.node_genes[0].bias_mean)
        self.assertEqual(parameters['layer_0.bias'][1], genome.node_genes[1].bias_mean)

        # input to hidden
        self.assertEqual(parameters['layer_1.bias'][0], genome.node_genes[2].bias_mean)
        self.assertEqual(parameters['layer_1.bias'][1], genome.node_genes[3].bias_mean)
        self.assertEqual(parameters['layer_1.bias'][2], genome.node_genes[4].bias_mean)


        self.assertEqual(parameters['layer_1.weight'][0, 0], genome.connection_genes[(-1, 2)].weight_mean)
        self.assertEqual(parameters['layer_1.weight'][0, 1], genome.connection_genes[(-2, 2)].weight_mean)
        self.assertEqual(parameters['layer_1.weight'][1, 0], genome.connection_genes[(-1, 3)].weight_mean)
        self.assertEqual(parameters['layer_1.weight'][1, 1], genome.connection_genes[(-2, 3)].weight_mean)
        self.assertEqual(parameters['layer_1.weight'][2, 0], genome.connection_genes[(-1, 4)].weight_mean)
        self.assertEqual(parameters['layer_1.weight'][2, 1], genome.connection_genes[(-2, 4)].weight_mean)
