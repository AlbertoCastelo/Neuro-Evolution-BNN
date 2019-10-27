from unittest import TestCase

import torch

from neat.gene import NodeGene, ConnectionGene
from neat.genome import Genome
from neat.representation_mapping.genome_to_network.complex_stochastic_network import transform_genome_to_layers, \
    ComplexStochasticNetwork
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
            self.assertEqual(layer['n_input'], 2)

        connections_0 = layers[0]['weight_mean']
        self.assertTrue(torch.allclose(connections_0, torch.tensor([[3.0, 4.0]]), atol=1e-02))

        connections_1 = layers[1]['weight_mean']
        self.assertTrue(torch.allclose(connections_1, torch.tensor([[0.0, 0.0], [1.0, 2.0]]), atol=1e-02))

    def test_network_structure_miso_2(self):

        graph = ((-1, 1), (-2, 1), (-1, 2), (-2, 2),
                 (1, 4), (1, 3), (2, 3), (2, 4),
                 (3, 0), (4, 0), (-1, 0), (1, 0))
        connection_weights = (1.0, 2.0, 3.0, 4.0,
                              5.0, 6.0, 7.0, 8.0,
                              1.0, 2.0, 3.0, 4.0)
        genome = generate_genome_given_graph(graph, connection_weights)

        layers = transform_genome_to_layers(genome)

        layer_0 = layers[0]
        self.assertEqual(len(layer_0['input_keys']), 4)
        self.assertEqual(len(layer_0['output_keys']), 1)
        print(layer_0['weight_mean'])
        self.assertTrue(torch.allclose(layer_0['weight_mean'], torch.tensor([[4.0, 1.0, 2.0, 3.0]]), atol=1e-02))

        layer_1 = layers[1]
        self.assertEqual(len(layer_1['input_keys']), 3)
        self.assertEqual(len(layer_1['output_keys']), 2)
        print(layer_1['weight_mean'])
        self.assertTrue(torch.allclose(layer_1['weight_mean'], torch.tensor([[1.0, 2.0, 3.0, 4.0]]), atol=1e-02))

        layer_2 = layers[2]
        self.assertEqual(len(layer_2['input_keys']), 2)
        self.assertEqual(len(layer_2['output_keys']), 2)
        print(layer_2['weight_mean'])
        self.assertTrue(torch.allclose(layer_2['weight_mean'], torch.tensor([[1.0, 2.0],
                                                                             [3.0, 4.0]]), atol=1e-02))


class TestStochasticFeedForwardWithMultiHopJumps(TestCase):

    def test_network_structure_miso(self):
        self.config = create_configuration(filename='/miso.json')
        genome = generate_genome_given_graph(graph=((-1, 1), (-2, 1), (1, 0), (-1, 0)),
                                             connection_weights=(1.0, 2.0, 3.0, 4.0))

        layers = transform_genome_to_layers(genome)

        layer_0 = layers[0]
        self.assertEqual(len(layer_0['input_keys']), 2)
        self.assertEqual(len(layer_0['output_keys']), 1)
        print(layer_0['weight_mean'])
        self.assertTrue(torch.allclose(layer_0['weight_mean'], torch.tensor([[3.0, 4.0]]), atol=1e-02))
