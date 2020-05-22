from unittest import TestCase, skip
import os
from config_files.configuration_utils import create_configuration
from neat.evolution_operators.mutation import ArchitectureMutation
from neat.genome import Genome
from neat.neat_logger import get_neat_logger
from tests.utils.generate_genome import generate_genome_with_hidden_units
LOGS_PATH = f'{os.getcwd()}/'
logger = get_neat_logger(path=LOGS_PATH)


class TestArchitectureMutationAddNode(TestCase):
    def test_mutate_add_node(self):
        pass


class TestArchitectureMutationDeleteConnection(TestCase):
    def setUp(self) -> None:
        self.config = create_configuration(filename='/regression-miso.json')
        self.config.n_initial_hidden_neurons = 0
        self.config.is_initial_fully_connected = True

    def test_mutate_delete_connection(self):
        genome = Genome(key='foo').create_random_genome()

        possible_connections_to_delete = \
            set(ArchitectureMutation._calculate_possible_connections_to_delete(genome=genome))
        expected_possible_connections_to_delete = {(-1, 0), (-2, 0)}
        self.assertSetEqual(expected_possible_connections_to_delete, possible_connections_to_delete)

    def test_mutate_delete_connection_when_two_input_one_hidden(self):
        self.config.n_initial_hidden_neurons = 1
        genome = Genome(key='foo').create_random_genome()

        possible_connections_to_delete = set(ArchitectureMutation._calculate_possible_connections_to_delete(genome=genome))
        expected_possible_connections_to_delete = {(-1, 1), (-2, 1), (1, 0)}
        self.assertSetEqual(expected_possible_connections_to_delete, possible_connections_to_delete)

    def test_mutate_delete_connection_when_one_input_one_hidden(self):
        self.config.n_initial_hidden_neurons = 1

        genome = Genome(key='foo').create_random_genome()

        possible_connections_to_delete = set(ArchitectureMutation._calculate_possible_connections_to_delete(genome=genome))
        expected_possible_connections_to_delete = {(-1, 1), (1, 0), (-2, 1)}
        self.assertSetEqual(expected_possible_connections_to_delete, possible_connections_to_delete)


class TestArchitectureMutation(TestCase):
    def setUp(self) -> None:
        self.config = create_configuration(filename='/regression-miso.json')
        self.config.n_initial_hidden_neurons = 0
        self.config.is_initial_fully_connected = True

    def test_calculate_possible_inputs_when_adding_connection(self):
        genome = generate_genome_with_hidden_units(n_input=self.config.n_input,
                                                   n_output=self.config.n_output,
                                                   n_hidden=3)
        out_node = genome.node_genes[0]

        possible_inputs = list(ArchitectureMutation._calculate_possible_inputs_when_adding_connection(genome=genome,
                                                                                                      out_node_key=out_node.key,
                                                                                                      config=self.config))
        expected_possible_inputs = [-2, -1]
        self.assertEqual(expected_possible_inputs, possible_inputs)

    def test_remove_connection_that_introduces_multihop_jumps(self):
        genome = generate_genome_with_hidden_units(n_input=self.config.n_input,
                                                   n_output=self.config.n_output,
                                                   n_hidden=3)
        possible_connection_set = {(1, 2), (2, 2), (2, 3)}

        possible_connection_set = \
            ArchitectureMutation._remove_connection_that_introduces_multihop_jumps(genome=genome,
                                                                                   possible_connection_set=possible_connection_set)
        expected_possible_connection_set = set()
        self.assertSetEqual(possible_connection_set, expected_possible_connection_set)

    def test_remove_connection_that_introduces_cycles(self):
        genome = generate_genome_with_hidden_units(n_input=self.config.n_input,
                                                   n_output=self.config.n_output,
                                                   n_hidden=3)
        possible_connection_set = {(1, 2), (2, 2), (2, 3)}

        possible_connection_set = \
            ArchitectureMutation._remove_connection_that_introduces_cycles(genome=genome,
                                                                           possible_connection_set=possible_connection_set)
        expected_possible_connection_set = {(1, 2), (2, 3)}
        self.assertSetEqual(possible_connection_set, expected_possible_connection_set)
