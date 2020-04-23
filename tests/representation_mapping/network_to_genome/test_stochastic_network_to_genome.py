from unittest import TestCase

import os

from config_files.configuration_utils import create_configuration
from neat.neat_logger import get_neat_logger
from neat.representation_mapping.genome_to_network.complex_stochastic_network import ComplexStochasticNetwork

from neat.representation_mapping.network_to_genome.stochastic_network_to_genome import \
    convert_stochastic_network_to_genome
from tests.representation_mapping.genome_to_network.test_complex_stochastic_network import generate_genome_given_graph

logger = get_neat_logger(path=f'{os.getcwd()}/')


class TestStochasticNetwork2Genome(TestCase):
    def test_genome_conversion_without_jumps(self):
        config = create_configuration(filename='/classification-miso.json')

        original_genome = generate_genome_given_graph(graph=((-1, 2), (-2, 2), (2, 0), (2, 1)),
                                                      connection_weights=(1.0, 2.0, 3.0, 0.5))

        network = ComplexStochasticNetwork(genome=original_genome)
        new_genome = convert_stochastic_network_to_genome(network=network, original_genome=original_genome)

        self.assertEqual(original_genome, new_genome)

    def test_genome_conversion_with_jumps_1(self):
        config = create_configuration(filename='/classification-miso.json')

        original_genome = generate_genome_given_graph(graph=((-1, 2), (-2, 2), (2, 0), (2, 1), (-1, 1)),
                                                      connection_weights=(1.0, 2.0, 3.0, 0.5, 0.75))

        network = ComplexStochasticNetwork(genome=original_genome)
        new_genome = convert_stochastic_network_to_genome(network=network, original_genome=original_genome)

        self.assertEqual(original_genome, new_genome)

    def test_genome_conversion_with_jumps_2(self):
        config = create_configuration(filename='/classification-miso.json')

        original_genome = generate_genome_given_graph(graph=((-1, 3), (-2, 3), (3, 2), (-1, 2), (2, 0),
                                                             (2, 1), (-1, 1), (-2, 3)),
                                                      connection_weights=(1.0, 2.0, 3.0, 0.5, 0.75,
                                                                          0.8, 0.6, 0.9))

        network = ComplexStochasticNetwork(genome=original_genome)
        new_genome = convert_stochastic_network_to_genome(network=network, original_genome=original_genome)

        self.assertEqual(original_genome, new_genome)

    def test_genome_conversion_fails_when_some_parameter_is_different(self):
        config = create_configuration(filename='/classification-miso.json')

        original_genome = generate_genome_given_graph(graph=((-1, 3), (-2, 3), (3, 2), (-1, 2), (2, 0),
                                                             (2, 1), (-1, 1), (-2, 3)),
                                                      connection_weights=(1.0, 2.0, 3.0, 0.5, 0.75,
                                                                          0.8, 0.6, 0.9))
        network = ComplexStochasticNetwork(genome=original_genome)
        network.layer_0.qw_mean[0, 1] = 0.33

        self.assertRaises(Exception, convert_stochastic_network_to_genome, network, original_genome)
