from unittest import TestCase

from neat.evolution_operators.mutation import Mutation, exist_cycle, adds_multihop_jump
from tests.config_files.config_files import create_configuration
from tests.utils.generate_genome import generate_genome_with_hidden_units


class TestMutation(TestCase):
    def test_calculate_possible_inputs(self):
        self.config = create_configuration(filename='/miso.json')
        genome = generate_genome_with_hidden_units(n_input=self.config.n_input,
                                                   n_output=self.config.n_output,
                                                   n_hidden=3)
        out_node = genome.node_genes[2]

        possible_inputs = list(Mutation._calculate_possible_inputs(genome=genome,
                                                                   out_node_key=out_node.key,
                                                                   config=self.config))
        expected_possible_inputs = []
        self.assertEqual(expected_possible_inputs, possible_inputs)

    def test_remove_connection_that_introduces_multihop_jumps(self):
        self.config = create_configuration(filename='/miso.json')
        genome = generate_genome_with_hidden_units(n_input=self.config.n_input,
                                                   n_output=self.config.n_output,
                                                   n_hidden=3)
        possible_connection_set = {(1, 2), (2, 2), (2, 3)}

        possible_connection_set = \
            Mutation._remove_connection_that_introduces_multihop_jumps(genome=genome,
                                                                       possible_connection_set=possible_connection_set)
        expected_possible_connection_set = set()
        self.assertSetEqual(possible_connection_set, expected_possible_connection_set)

    def test_remove_connection_that_introduces_cycles(self):
        self.config = create_configuration(filename='/miso.json')
        genome = generate_genome_with_hidden_units(n_input=self.config.n_input,
                                                   n_output=self.config.n_output,
                                                   n_hidden=3)
        possible_connection_set = {(1, 2), (2, 2), (2, 3)}

        possible_connection_set = \
            Mutation._remove_connection_that_introduces_cycles(genome=genome,
                                                               possible_connection_set=possible_connection_set)
        expected_possible_connection_set = {(1, 2), (2, 3)}
        self.assertSetEqual(possible_connection_set, expected_possible_connection_set)


class TestMultihopJumps(TestCase):
    def test_adds_multihop_jump_true_a(self):
        connections = [(-1, 1), (-1, 2), (-1, 3), (-2, 1), (-2, 2), (-2, 3), (1, 0), (2, 0), (3, 0), (1, 2)]
        self.assertEqual(True, adds_multihop_jump(connections=connections,
                                                  output_node_keys=[0],
                                                  input_node_keys=[-1, -2]))

    def test_adds_multihop_jump_true_b(self):
        connections = [(-1, 1), (-2, 1), (1, 2), (2, 3), (2, 0), (3, 0)]
        self.assertEqual(True, adds_multihop_jump(connections=connections,
                                                  output_node_keys=[0],
                                                  input_node_keys=[-1, -2]))

    def test_adds_multihop_jump_false_a(self):
        connections = [(-1, 1), (-1, 2), (-1, 3), (-2, 1), (-2, 2), (-2, 3), (1, 0), (2, 0), (3, 0)]
        self.assertEqual(False, adds_multihop_jump(connections=connections,
                                                   output_node_keys=[0],
                                                   input_node_keys=[-1, -2]))


class TestExistCycles(TestCase):
    def test_exists_cycle_positive_a(self):
        connections = [(1, 2), (2, 3), (3, 1)]

        self.assertEqual(True, exist_cycle(connections))

    def test_exists_cycle_positive_b(self):
        connections = [(1, 2), (2, 3), (1, 4), (4, 3), (3, 1)]

        self.assertEqual(True, exist_cycle(connections))

    def test_exists_cycle_when_negative_a(self):
        connections = [(1, 2), (2, 3), (3, 4)]

        self.assertEqual(False, exist_cycle(connections))

    def test_exists_cycle_when_negative_b(self):
        connections = [(1, 2), (2, 3), (1, 4), (4, 3), (3, 5)]

        self.assertEqual(False, exist_cycle(connections))

    def test_exists_cycle_when_negative_c(self):
        connections = [(-1, 1), (-1, 2), (-1, 3), (-2, 1), (-2, 2), (-2, 3), (1, 0), (2, 0), (3, 0), (1, 2)]
        self.assertEqual(False, exist_cycle(connections))

    def test_self_recursive(self):
        connections = [(1, 1)]

        self.assertEqual(True, exist_cycle(connections))

    def test_self_recursive_b(self):
        connections = [(-1, 1), (-1, 2), (-1, 3), (-2, 1),  (-2, 2), (-2, 3), (1, 0), (2, 0), (3, 0), (2, 2)]
        self.assertEqual(True, exist_cycle(connections))
