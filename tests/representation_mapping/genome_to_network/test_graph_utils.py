from unittest import TestCase
from neat.representation_mapping.genome_to_network.graph_utils import calculate_nodes_per_layer, \
    calculate_max_graph_depth_per_node


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
