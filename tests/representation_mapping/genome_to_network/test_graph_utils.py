from unittest import TestCase

from neat.neat_logger import get_neat_logger
from neat.representation_mapping.genome_to_network.graph_utils import calculate_nodes_per_layer, \
    calculate_max_graph_depth_per_node, adds_multihop_jump, exist_cycle, exist_cycle_numba
import os

LOGS_PATH = f'{os.getcwd()}/'
logger = get_neat_logger(path=LOGS_PATH)


class TestCalculateNodesPerLayer(TestCase):
    def test_case_1(self):
        links = ((-1, 1), (-2, 1), (1, 0))
        nodes_per_layer = calculate_nodes_per_layer(links=links,
                                                    output_node_keys=[0],
                                                    input_node_keys=[-1, -2])

        expected_nodes_per_layer = {0: [0],
                                    1: [1],
                                    2: [-1, -2]}
        self.assertEqual(nodes_per_layer, expected_nodes_per_layer)

    def test_case_2(self):
        links = ((-1, 1), (-2, 1), (-1, 0), (1, 0))
        nodes_per_layer = calculate_nodes_per_layer(links=links,
                                                    output_node_keys=[0],
                                                    input_node_keys=[-1, -2])

        expected_nodes_per_layer = {0: [0],
                                    1: [1],
                                    2: [-1, -2]}
        self.assertEqual(nodes_per_layer, expected_nodes_per_layer)

    def test_case_3(self):
        links = ((-1, 1), (-2, 1), (-1, 0), (-2, 0), (1, 0))
        nodes_per_layer = calculate_nodes_per_layer(links=links,
                                                    output_node_keys=[0],
                                                    input_node_keys=[-1, -2])

        expected_nodes_per_layer = {0: [0],
                                    1: [1],
                                    2: [-1, -2]}
        self.assertEqual(nodes_per_layer, expected_nodes_per_layer)

    def test_case_4(self):
        links = ((-1, 1), (-2, 0), (-2, 1), (-1, 2), (2, 0), (-1, 0))
        nodes_per_layer = calculate_nodes_per_layer(links=links,
                                                    output_node_keys=[0, 1],
                                                    input_node_keys=[-1, -2])

        expected_nodes_per_layer = {0: [0, 1],
                                    1: [2],
                                    2: [-1, -2]}
        self.assertEqual(nodes_per_layer, expected_nodes_per_layer)

    def test_case_5(self):
        links = ((-1, 0), (-1, 1), (-2, 0), (-2, 2), (2, 1))
        nodes_per_layer = calculate_nodes_per_layer(links=links,
                                                    output_node_keys=[0, 1],
                                                    input_node_keys=[-1, -2])

        expected_nodes_per_layer = {0: [0, 1],
                                    1: [2],
                                    2: [-1, -2]}
        self.assertEqual(nodes_per_layer, expected_nodes_per_layer)

    def test_case_6(self):
        links = ((-1, 1), (-2, 1), (-1, 2), (-2, 2),
                 (1, 4), (1, 3), (2, 3), (2, 4), (-1, 3),
                 (3, 0), (4, 0), (-1, 0), (1, 0))

        nodes_per_layer = calculate_nodes_per_layer(links=links,
                                                    output_node_keys=[0],
                                                    input_node_keys=[-1, -2])
        expected_nodes_per_layer = {0: [0],
                                    1: [3, 4],
                                    2: [1, 2],
                                    3: [-1, -2]}
        self.assertEqual(nodes_per_layer, expected_nodes_per_layer)

    def test_case_7(self):
        links = ((-1, 1), (-2, 1))
        nodes_per_layer = calculate_nodes_per_layer(links=links,
                                                    output_node_keys=[0, 1],
                                                    input_node_keys=[-1, -2])
        expected_nodes_per_layer = {0: [0, 1],
                                    1: [-1, -2]}
        self.assertEqual(nodes_per_layer, expected_nodes_per_layer)

    def test_case_8(self):
        links = ((-1, 0), (-1, 1), (-2, 0), (2, 1), (2, 0), (-2, 3), (3, 2))
        nodes_per_layer = calculate_nodes_per_layer(links=links,
                                                    output_node_keys=[0, 1],
                                                    input_node_keys=[-1, -2])
        expected_nodes_per_layer = {0: [0, 1],
                                    1: [2],
                                    2: [3],
                                    3: [-1, -2]}
        self.assertEqual(nodes_per_layer, expected_nodes_per_layer)

    def test_case_9(self):
        links = ((-2, 1), (-1, 0), (-2, 0), (-1, 2), (2, 3), (3, 1), (3, 0))
        nodes_per_layer = calculate_nodes_per_layer(links=links,
                                                    output_node_keys=[0, 1],
                                                    input_node_keys=[-1, -2])
        expected_nodes_per_layer = {0: [0, 1],
                                    1: [3],
                                    2: [2],
                                    3: [-1, -2]}
        self.assertEqual(expected_nodes_per_layer, nodes_per_layer)


class TestMaxGraphDepthPerNode(TestCase):
    def test_simple_case(self):
        links = ((-1, 1), (-2, 1), (1, 0))
        max_graph_depth_per_node = calculate_max_graph_depth_per_node(links=links)
        print(max_graph_depth_per_node)
        expected_max_depth = {0: 2,
                              1: 1,
                              -1: 0,
                              -2: 0}
        self.assertEqual(expected_max_depth, max_graph_depth_per_node)

    def test_with_jumps(self):
        links = ((-1, 1), (-2, 1), (-1, 2), (-2, 2),
                 (1, 4), (1, 3), (2, 3), (2, 4), (-1, 3),
                 (3, 0), (4, 0), (-1, 0), (1, 0))
        max_graph_depth_per_node = calculate_max_graph_depth_per_node(links=links)
        print(max_graph_depth_per_node)
        expected_max_depth = {0: 3,
                              3: 2,
                              4: 2,
                              1: 1,
                              2: 1,
                              -1: 0,
                              -2: 0}
        self.assertEqual(expected_max_depth, max_graph_depth_per_node)



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


class TestExistCyclesNumba(TestCase):
    def test_exists_cycle_positive_a(self):
        connections = [(1, 2), (2, 3), (3, 1)]

        self.assertEqual(True, exist_cycle_numba(connections))

    def test_exists_cycle_positive_b(self):
        connections = [(1, 2), (2, 3), (1, 4), (4, 3), (3, 1)]

        self.assertEqual(True, exist_cycle_numba(connections))

    def test_exists_cycle_when_negative_a(self):
        connections = [(1, 2), (2, 3), (3, 4)]

        self.assertEqual(False, exist_cycle_numba(connections))

    def test_exists_cycle_when_negative_b(self):
        connections = [(1, 2), (2, 3), (1, 4), (4, 3), (3, 5)]

        self.assertEqual(False, exist_cycle_numba(connections))

    def test_exists_cycle_when_negative_c(self):
        connections = [(-1, 1), (-1, 2), (-1, 3), (-2, 1), (-2, 2), (-2, 3), (1, 0), (2, 0), (3, 0), (1, 2)]
        self.assertEqual(False, exist_cycle_numba(connections))

    def test_self_recursive(self):
        connections = [(1, 1)]

        self.assertEqual(True, exist_cycle_numba(connections))

    def test_self_recursive_b(self):
        connections = [(-1, 1), (-1, 2), (-1, 3), (-2, 1),  (-2, 2), (-2, 3), (1, 0), (2, 0), (3, 0), (2, 2)]
        self.assertEqual(True, exist_cycle_numba(connections))

