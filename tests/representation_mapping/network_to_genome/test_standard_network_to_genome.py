from unittest import TestCase

import torch
from neat.genome import Genome
from neat.representation_mapping.network_to_genome.standard_feed_forward_to_genome import \
    get_genome_from_standard_network
from config_files import create_configuration
from deep_learning.standard.feed_forward import FeedForward


class TestNetwork2Genome(TestCase):
    def test_mapping(self):
        config = create_configuration(filename='/classification-miso.json')
        model_filename = 'network-classification.pt'
        n_neurons_per_layer = 10
        network = FeedForward(n_input=config.n_input, n_output=config.n_output,
                              n_neurons_per_layer=n_neurons_per_layer,
                              n_hidden_layers=1)
        # print(os.getcwd())
        parameters = torch.load(f'./../../../tests_non_automated/deep_learning/models/{model_filename}')
        network.load_state_dict(parameters)

        std = 0.1
        genome = get_genome_from_standard_network(network, std=std)

        self.assertEqual(type(genome), Genome)
