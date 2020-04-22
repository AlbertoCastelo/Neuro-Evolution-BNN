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
    def test_mapping(self):
        config = create_configuration(filename='/classification-miso.json')

        original_genome = generate_genome_given_graph(graph=((-1, 2), (-2, 2), (2, 0), (2, 1)),
                                                      connection_weights=(1.0, 2.0, 3.0, 0.5))

        network = ComplexStochasticNetwork(genome=original_genome)
        new_genome = convert_stochastic_network_to_genome(network=network, original_genome=original_genome)

        self.assertEqual(original_genome, new_genome)
