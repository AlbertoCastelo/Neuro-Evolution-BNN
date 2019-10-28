from unittest import TestCase

import torch

from neat.gene import NodeGene, ConnectionGene
from neat.genome import Genome
from neat.representation_mapping.genome_to_network.complex_stochastic_network import transform_genome_to_layers, \
    ComplexStochasticNetwork, calculate_max_graph_depth_per_node
from neat.representation_mapping.genome_to_network.stochastic_network import StochasticNetworkOld
from tests.config_files.config_files import create_configuration

STD = 0.00000001


def generate_genome_given_graph(graph, connection_weights):
    unique_node_keys = []
    for connection in graph:
        for node_key in connection:
            if node_key not in unique_node_keys:
                unique_node_keys.append(node_key)

    nodes = {}
    for node_key in unique_node_keys:
        node = NodeGene(key=node_key).random_initialization()
        node.set_mean(0)
        node.set_std(STD)
        nodes[node_key] = node

    connections = {}
    for connection_key, weight in zip(graph, connection_weights):
        connection = ConnectionGene(key=connection_key)
        connection.set_mean(weight)
        connection.set_std(STD)
        connections[connection_key] = connection

    genome = Genome(key='foo')
    genome.connection_genes = connections
    genome.node_genes = nodes
    return genome


class TestTransformGenomeWithMultiHopJumpsToLayers(TestCase):
    def setUp(self) -> None:
        self.config = create_configuration(filename='/miso.json')

    def test_network_structure_miso(self):
        graph = ((-1, 1), (-2, 1), (1, 0), (-1, 0))
        connection_weights = (1.0, 2.0, 3.0, 4.0)
        genome = generate_genome_given_graph(graph, connection_weights)

        layers = transform_genome_to_layers(genome)

        # we have 2 layers
        self.assertEqual(len(layers), 2)

        # we have 2 nodes in each layer
        for layer in layers.values():
            self.assertEqual(layer.n_input, 2)

        connections_0 = layers[0].weight_mean
        print(connections_0)
        self.assertTrue(torch.allclose(connections_0, torch.tensor([[3.0, 4.0]]), atol=1e-02))

        connections_1 = layers[1].weight_mean
        print(connections_1)
        self.assertTrue(torch.allclose(connections_1, torch.tensor([[2.0, 1.0]]), atol=1e-02))

    def test_network_structure_miso_2(self):

        graph = ((-1, 1), (-2, 1), (-1, 2), (-2, 2),
                 (1, 4), (1, 3), (2, 3), (2, 4), (-1, 3),
                 (3, 0), (4, 0), (-1, 0), (1, 0))
        connection_weights = (1.0, 2.0, 3.0, 4.0,
                              5.0, 6.0, 7.0, 8.0, 1.5,
                              1.0, 2.0, 3.0, 4.0)
        genome = generate_genome_given_graph(graph, connection_weights)

        layers = transform_genome_to_layers(genome)

        layer_0 = layers[0]
        self.assertEqual(len(layer_0.input_keys), 4)
        self.assertEqual(len(layer_0.output_keys), 1)
        print(layer_0.weight_mean)
        self.assertTrue(torch.allclose(layer_0.weight_mean, torch.tensor([[1.0, 2.0, 3.0, 4.0]]), atol=1e-02))

        layer_1 = layers[1]
        self.assertEqual(len(layer_1.input_keys), 3)
        self.assertEqual(len(layer_1.output_keys), 2)
        print(layer_1.weight_mean)
        self.assertTrue(torch.allclose(layer_1.weight_mean, torch.tensor([[6.0000, 7.0000, 1.5000],
                                                                          [5.0000, 8.0000, 0.0000]]), atol=1e-02))

        layer_2 = layers[2]
        self.assertEqual(len(layer_2.input_keys), 2)
        self.assertEqual(len(layer_2.output_keys), 2)
        print(layer_2.weight_mean)
        self.assertTrue(torch.allclose(layer_2.weight_mean, torch.tensor([[2.0, 1.0],
                                                                          [4.0, 3.0]]), atol=1e-02))


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


class TestComplexStochasticNetwork(TestCase):

    def test_network_structure_miso(self):
        self.config = create_configuration(filename='/miso.json')
        self.config.node_activation = 'identity'
        genome = generate_genome_given_graph(graph=((-1, 1), (-2, 1), (1, 0), (-1, 0)),
                                             connection_weights=(1.0, 2.0, 3.0, 4.0))
        n_samples = 1
        input_data = torch.tensor([[1.0, 1.0]])
        input_data = input_data.view(-1, genome.n_input).repeat(n_samples, 1)

        model = ComplexStochasticNetwork(genome=genome)

        y, _ = model(input_data)

        expected_y = 13.0
        self.assertAlmostEqual(expected_y, y.numpy()[0][0])

    def test_network_structure_miso_2(self):
        self.config = create_configuration(filename='/miso.json')
        self.config.node_activation = 'identity'
        self.config.n_output = 2
        genome = generate_genome_given_graph(graph=((-1, 2), (-2, 2), (2, 0), (2, 1),
                                                    (-1, 0), (-1, 1), (-2, 1), (-2, 0)),
                                             connection_weights=(1.0, 2.0, 3.0, 4.0,
                                                                 0, 1.0, 0, 1.0))
        n_samples = 1
        input_data = torch.tensor([[1.0, 1.0]])
        input_data = input_data.view(-1, genome.n_input).repeat(n_samples, 1)

        model = ComplexStochasticNetwork(genome=genome)
        # print(model.layers[0].weight_mean)
        self.assertEqual(model.layers[0].input_keys, [2, -2, -1])
        self.assertTrue(torch.allclose(model.layers[0].weight_mean,
                                       torch.tensor([[3.0, 1.0, 0.0],
                                                     [4.0, 0.0, 1.0]]), atol=1e-02))
        self.assertEqual(model.layers[1].input_keys, [-2, -1])
        self.assertTrue(torch.allclose(model.layers[1].weight_mean,
                                       torch.tensor([[2.0, 1.0]]), atol=1e-02))

        y, _ = model(input_data)

        expected_y = torch.tensor([[10.0, 13.0]])
        self.assertTrue(torch.allclose(y, expected_y, atol=1e-02))

    def test_network_structure_miso_3(self):
        self.config = create_configuration(filename='/miso.json')
        self.config.node_activation = 'identity'

        graph = ((-1, 1), (-2, 1), (-1, 2), (-2, 2),
                 (1, 4), (1, 3), (2, 3), (2, 4), (-1, 3),
                 (3, 0), (4, 0), (-1, 0), (1, 0))
        connection_weights = (1.0, 2.0, 3.0, 4.0,
                              5.0, 6.0, 7.0, 8.0, 1.5,
                              1.0, 2.0, 3.0, 4.0)
        genome = generate_genome_given_graph(graph, connection_weights)

        n_samples = 100
        input_data = torch.tensor([[1.0, 1.0]])
        input_data = input_data.view(-1, genome.n_input).repeat(n_samples, 1)

        model = ComplexStochasticNetwork(genome=genome)

        y, _ = model(input_data)

        expected_y = 225
        self.assertAlmostEqual(expected_y, y.mean().item(), delta=10)