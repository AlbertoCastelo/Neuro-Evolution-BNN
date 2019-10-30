from unittest import TestCase

import torch

from neat.gene import NodeGene, ConnectionGene
from neat.genome import Genome
from neat.representation_mapping.genome_to_network.complex_stochastic_network import transform_genome_to_layers, \
    ComplexStochasticNetwork
from tests.config_files.config_files import create_configuration

STD = 0.00000001


class TestTransformGenomeWithMultiHopJumpsToLayers(TestCase):
    def setUp(self) -> None:
        self.config = create_configuration(filename='/miso.json')

    def test_network_structure_0(self):
        '''
        1 layers
        '''
        self.config = create_configuration(filename='/miso.json')
        self.config.node_activation = 'identity'
        genome = generate_genome_given_graph(graph=((-1, 0), (-2, 0)),
                                             connection_weights=(1.0, 2.0))

        layers = transform_genome_to_layers(genome)

        self.assertEqual(layers[0].input_keys, [-2, -1])
        self.assertEqual(layers[0].indices_of_needed_nodes, [])
        self.assertEqual(layers[0].indices_of_nodes_to_cache, [])
        self.assertTrue(torch.allclose(layers[0].weight_mean,
                                       torch.tensor([[2.0, 1.0]]), atol=1e-02))

    def test_network_structure_1(self):
        '''
        2 layers
        '''
        self.config = create_configuration(filename='/miso.json')
        self.config.node_activation = 'identity'
        genome = generate_genome_given_graph(graph=((-1, 1), (-2, 1), (1, 0)),
                                             connection_weights=(1.0, 2.0, 3.0))

        layers = transform_genome_to_layers(genome)

        self.assertEqual(layers[0].input_keys, [1])
        self.assertEqual(layers[0].indices_of_needed_nodes, [])
        self.assertEqual(layers[0].indices_of_nodes_to_cache, [])
        self.assertTrue(torch.allclose(layers[0].weight_mean,
                                       torch.tensor([[3.0]]), atol=1e-02))

        self.assertEqual(layers[1].input_keys, [-2, -1])
        self.assertEqual(layers[1].indices_of_needed_nodes, [])
        self.assertEqual(layers[1].indices_of_nodes_to_cache, [])
        self.assertTrue(torch.allclose(layers[1].weight_mean,
                                       torch.tensor([[2.0, 1.0]]), atol=1e-02))

    def test_network_structure_2(self):
        '''
        1 connection jumping one layer
        '''
        graph = ((-1, 1), (-2, 1), (1, 0), (-1, 0))
        connection_weights = (1.0, 2.0, 3.0, 4.0)
        genome = generate_genome_given_graph(graph, connection_weights)

        layers = transform_genome_to_layers(genome)
        self.assertEqual(len(layers), 2)

        self.assertEqual(layers[0].input_keys, [1, -1])
        self.assertEqual(layers[0].indices_of_needed_nodes, [(1, 1)])
        self.assertEqual(layers[0].indices_of_nodes_to_cache, [])
        connections_0 = layers[0].weight_mean
        self.assertTrue(torch.allclose(connections_0, torch.tensor([[3.0, 4.0]]), atol=1e-02))

        self.assertEqual(layers[1].input_keys, [-2, -1])
        self.assertEqual(layers[1].indices_of_needed_nodes, [])
        self.assertEqual(layers[1].indices_of_nodes_to_cache, [1])
        connections_1 = layers[1].weight_mean
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
        self.assertEqual(layers[0].indices_of_needed_nodes, [(2, 1), (1, 0)])
        self.assertEqual(layers[0].indices_of_nodes_to_cache, [])
        self.assertTrue(torch.allclose(layer_0.weight_mean, torch.tensor([[1.0, 2.0, 3.0, 4.0]]), atol=1e-02))

        layer_1 = layers[1]
        self.assertEqual(layer_1.input_keys, [1, 2, -1])
        self.assertEqual(layer_1.output_keys, [3, 4])
        self.assertEqual(layers[1].indices_of_needed_nodes, [(2, 1)])
        self.assertEqual(layers[1].indices_of_nodes_to_cache, [0])
        self.assertTrue(torch.allclose(layer_1.weight_mean, torch.tensor([[6.0000, 7.0000, 1.5000],
                                                                          [5.0000, 8.0000, 0.0000]]), atol=1e-02))

        layer_2 = layers[2]
        self.assertEqual(layer_2.input_keys, [-2, -1])
        self.assertEqual(layer_2.output_keys, [1, 2])
        self.assertEqual(layer_2.indices_of_needed_nodes, [])
        self.assertEqual(layer_2.indices_of_nodes_to_cache, [1])
        self.assertTrue(torch.allclose(layer_2.weight_mean, torch.tensor([[2.0, 1.0],
                                                                          [4.0, 3.0]]), atol=1e-02))

    def test_network_structure_5(self):
        self.config = create_configuration(filename='/miso.json')
        self.config.node_activation = 'identity'
        self.config.n_output = 2
        graph = ((-1, 1), (-2, 0), (-2, 1), (-1, 2), (2, 0), (-1, 0))
        weights = (1, 1, 1, 1, 1, 1)

        genome = generate_genome_given_graph(graph, weights)
        layers = transform_genome_to_layers(genome=genome)

        layer_0 = layers[0]
        self.assertEqual(layer_0.input_keys, [2, -2, -1])
        self.assertEqual(layer_0.output_keys, [0, 1])
        self.assertEqual(layer_0.indices_of_needed_nodes, [(1, 0), (1, 1)])
        self.assertEqual(layer_0.indices_of_nodes_to_cache, [])
        self.assertTrue(torch.allclose(layer_0.weight_mean, torch.tensor([[1.0, 1.0, 1.0],
                                                                          [0.0, 1.0, 1.0]]), atol=1e-02))

        layer_1 = layers[1]
        self.assertEqual(layer_1.input_keys, [-2, -1])
        self.assertEqual(layer_1.output_keys, [2])
        self.assertEqual(layer_1.indices_of_needed_nodes, [])
        self.assertEqual(layer_1.indices_of_nodes_to_cache, [0, 1])
        self.assertTrue(torch.allclose(layer_1.weight_mean, torch.tensor([[0.0, 1.0]]), atol=1e-02))

    def test_network_structure_when_one_output_is_not_connected(self):
        self.config.n_output = 2
        graph = ((-1, 1), (-2, 1))
        weights = (1, 1)
        genome = generate_genome_given_graph(graph, weights)
        layers = transform_genome_to_layers(genome=genome)

        layer_0 = layers[0]
        self.assertEqual(layer_0.input_keys, [-2, -1])
        self.assertEqual(layer_0.output_keys, [0, 1])
        self.assertEqual(layer_0.indices_of_needed_nodes, [])
        self.assertEqual(layer_0.indices_of_nodes_to_cache, [])
        self.assertTrue(torch.allclose(layer_0.weight_mean, torch.tensor([[0, 0],
                                                                          [1.0, 1.0]]), atol=1e-02))

    def test_network_structure_7(self):
        self.config.n_output = 2

        graph = ((-1, 0), (-1, 1), (-2, 0), (2, 1), (2, 0), (-2, 3), (3, 2))

        weights = (1, 1, 1, 1, 1, 1, 1)
        genome = generate_genome_given_graph(graph, weights)
        layers = transform_genome_to_layers(genome=genome)
        self.assertEqual(3, len(layers))

        layer_0 = layers[0]
        self.assertEqual(layer_0.input_keys, [2, -2, -1])
        self.assertEqual(layer_0.output_keys, [0, 1])
        self.assertEqual(layer_0.indices_of_needed_nodes, [(2, 0), (2, 1)])
        self.assertEqual(layer_0.indices_of_nodes_to_cache, [])
        self.assertTrue(torch.allclose(layer_0.weight_mean, torch.tensor([[1.0, 1.0, 1.0],
                                                                          [0.0, 1.0, 0.0]]), atol=1e-02))


class TestComplexStochasticNetwork(TestCase):
    def setUp(self) -> None:
        self.config = create_configuration(filename='/miso.json')
        self.config.node_activation = 'identity'

    def test_network_structure_1(self):
        genome = generate_genome_given_graph(graph=((-1, 1), (-2, 1), (1, 0)),
                                             connection_weights=(1.0, 2.0, 3.0))
        n_samples = 1
        input_data = torch.tensor([[1.0, 1.0]])
        input_data = input_data.view(-1, genome.n_input).repeat(n_samples, 1)

        model = ComplexStochasticNetwork(genome=genome)
        self.assertEqual(model.layers[0].input_keys, [1])
        self.assertTrue(torch.allclose(model.layers[0].weight_mean,
                                       torch.tensor([[3.0]]), atol=1e-02))

        self.assertEqual(model.layers[1].input_keys, [-2, -1])
        self.assertTrue(torch.allclose(model.layers[1].weight_mean,
                                       torch.tensor([[2.0, 1.0]]), atol=1e-02))

        y, _ = model(input_data)

        expected_y = 9.0
        self.assertAlmostEqual(expected_y, y.numpy()[0][0])

    def test_network_structure_miso(self):
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

    def test_network_structure_mimo_4(self):
        self.config.n_output = 2
        graph = ((-1, 0), (-1, 1), (-2, 0), (-2, 2), (2, 1))
        weights = (1, 1, 1, 1, 1)

        genome = generate_genome_given_graph(graph, weights)
        model = ComplexStochasticNetwork(genome=genome)

        n_samples = 1
        input_data = torch.tensor([[1.0, 1.0]])
        input_data = input_data.view(-1, genome.n_input).repeat(n_samples, 1)
        y, _ = model(input_data)

    def test_network_structure_mimo_5(self):
        self.config.n_output = 2
        graph = ((-1, 1), (-2, 0), (-2, 1), (-1, 2), (2, 0), (-1, 0))
        weights = (1, 1, 1, 1, 1, 1)

        genome = generate_genome_given_graph(graph, weights)
        model = ComplexStochasticNetwork(genome=genome)

        n_samples = 1
        input_data = torch.tensor([[1.0, 1.0]])
        input_data = input_data.view(-1, genome.n_input).repeat(n_samples, 1)
        y, _ = model(input_data)

    def test_network_structure_without_all_output(self):
        self.config.n_output = 2
        graph = ((-1, 1), (-2, 1))
        weights = (1, 1)
        genome = generate_genome_given_graph(graph, weights)
        model = ComplexStochasticNetwork(genome=genome)
        n_samples = 1
        input_data = torch.tensor([[1.0, 1.0]])
        input_data = input_data.view(-1, genome.n_input).repeat(n_samples, 1)
        y, _ = model(input_data)


def generate_genome_given_graph(graph, connection_weights):
    genome = Genome(key='foo')

    unique_node_keys = []
    for connection in graph:
        for node_key in connection:
            if node_key not in unique_node_keys:
                unique_node_keys.append(node_key)

    unique_node_keys = list(set(unique_node_keys + genome.get_output_nodes_keys()))
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


    genome.connection_genes = connections
    genome.node_genes = nodes
    return genome
