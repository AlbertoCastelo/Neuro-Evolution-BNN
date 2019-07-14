from unittest import TestCase

from neat.gene import NodeGene
from neat.genome import GenomeSample
from neat.representation import Network
import torch


class TestRepresentationFeedForwardWithoutHiddenLayers(TestCase):

    def test_network_structure(self):
        node_0 = NodeGene(key=0)
        node_0.random_initialization()
        node_1 = NodeGene(key=1)
        node_1.random_initialization()

        node_genes = {0: node_0, 1: node_1}

        connection_genes = {(-1, 0): 1.5,
                            (-2, 0): 2.5,
                            (-1, 1): -0.5,
                            (-2, 1): 0.0}
        n_output = 2
        genome_sample = GenomeSample(key=None, n_input=2, n_output=n_output,
                                     node_genes=node_genes,
                                     connection_genes=connection_genes)

        model = Network(genome=genome_sample)

        input = torch.tensor([1.0, -1.0])
        result = model.forward(input.data)

        self.assertEqual(len(result), n_output)

    def test_connection_is_not_specified_assumes_0(self):
        node_0 = NodeGene(key=0)
        node_0.random_initialization()
        node_1 = NodeGene(key=1)
        node_1.random_initialization()

        node_genes = {0: node_0, 1: node_1}

        connection_genes = {(-1, 0): 1.5,
                            (-2, 0): 2.5,
                            (-1, 1): -0.5}
        n_output = 2
        genome_sample = GenomeSample(key=None, n_input=2, n_output=n_output,
                                     node_genes=node_genes,
                                     connection_genes=connection_genes)

        model = Network(genome=genome_sample)

        input = torch.tensor([1.0, -1.0])
        result = model.forward(input.data)
        missing_weight = model.state_dict()['layer_0.weight'].data[1, 1]
        self.assertEqual(0.0, missing_weight)
        self.assertEqual(len(result), n_output)

    def test_weights_are_correctly_set(self):
        node_0 = NodeGene(key=0)
        node_0.random_initialization()
        node_0.bias = 0.0
        node_1 = NodeGene(key=1)
        node_1.random_initialization()
        node_1.bias = 0.0

        node_genes = {0: node_0, 1: node_1}

        connection_genes = {(-1, 0): 1.5,
                            (-2, 0): 2.5,
                            (-1, 1): 0.0,
                            (-2, 1): 0.0}
        n_output = 2
        genome_sample = GenomeSample(key=None, n_input=2, n_output=n_output,
                                     node_genes=node_genes,
                                     connection_genes=connection_genes)

        # model = get_nn_from_genome(genome=genome_sample)
        model = Network(genome=genome_sample)
        input = torch.tensor([1.0, 1.0])
        result = model.forward(input.data)

        self.assertEqual(len(result), n_output)

        self.assertTrue(torch.allclose(result, torch.tensor([0.9820, 0.5]), atol=1e-02))

    def test_bias_are_correctly_set(self):
        node_0 = NodeGene(key=0)
        node_0.random_initialization()
        node_0.bias = 1.0
        node_1 = NodeGene(key=1)
        node_1.random_initialization()
        node_1.bias = 0.0
        node_genes = {0: node_0, 1: node_1}

        connection_genes = {(-1, 0): 0.0,
                            (-2, 0): 0.0,
                            (-1, 1): 0.0,
                            (-2, 1): 0.0}
        n_output = 2
        genome_sample = GenomeSample(key=None, n_input=2, n_output=n_output,
                                     node_genes=node_genes,
                                     connection_genes=connection_genes)

        # model = get_nn_from_genome(genome=genome_sample)
        model = Network(genome=genome_sample)
        input = torch.tensor([1.0, 1.0])
        result = model.forward(input.data)

        self.assertEqual(len(result), n_output)
        self.assertTrue(torch.allclose(result, torch.tensor([0.7310, 0.5]), atol=1e-03))


class TestRepresentationFeedForwardWithOneHiddenLayers(TestCase):

    def test_network_structure(self):
        genome_sample = generate_feedforward_with_one_hidden_unit()

        model = Network(genome=genome_sample)

        input = torch.tensor([1.0, -1.0])
        result = model.forward(input.data)

        self.assertEqual(len(result), genome_sample.n_output)
        self.assertEqual(model.n_layers, 2)
        self.assertTrue(torch.allclose(result, torch.tensor([0.9173, 0.7311]), atol=1e-02))

    def test_weights_are_located_correctly(self):
        genome_sample = generate_feedforward_with_one_hidden_unit()

        model = Network(genome=genome_sample)

        expected_bias_layer_0 = torch.tensor([1.0, 1.0])
        self.assertTrue(torch.allclose(model.layer_0.bias, expected_bias_layer_0))

        expected_bias_layer_1 = torch.tensor([1.0, -1.0, 0.0])
        self.assertTrue(torch.allclose(model.layer_1.bias, expected_bias_layer_1))

        expected_weight_layer_0 = torch.tensor([[0.0, -0.5, 1.5],
                                                [0.0, 0.0, 0.0]])
        self.assertTrue(torch.allclose(model.layer_0.weight, expected_weight_layer_0))

        expected_weight_layer_1 = torch.tensor([[1.5, 2.5],
                                                [-0.5, 0.0],
                                                [5.5, -1]])
        self.assertTrue(torch.allclose(model.layer_1.weight, expected_weight_layer_1))


def generate_feedforward_with_one_hidden_unit():
    # output nodes
    node_0 = NodeGene(key=0)
    node_0.random_initialization()
    node_0.bias = 1.0

    node_1 = NodeGene(key=1)
    node_1.random_initialization()
    node_1.bias = 1.0

    # hidden nodes
    node_2 = NodeGene(key=2)
    node_2.random_initialization()
    node_2.bias = 1.0

    node_3 = NodeGene(key=3)
    node_3.random_initialization()
    node_3.bias = -1.0

    node_4 = NodeGene(key=4)
    node_4.random_initialization()
    node_4.bias = 0.0

    node_genes = {0: node_0, 1: node_1, 2: node_2, 3: node_3, 4: node_4}

    connection_genes = {(-1, 2): 1.5,
                        (-2, 2): 2.5,
                        (-1, 3): -0.5,
                        (-2, 3): 0.0,
                        (-1, 4): 5.5,
                        (-2, 4): -1,
                        (3, 0): -0.5,
                        (2, 1): 0.0,
                        (4, 0): 1.5}

    n_output = 2
    genome_sample = GenomeSample(key=None, n_input=2, n_output=n_output,
                                 node_genes=node_genes,
                                 connection_genes=connection_genes)
    return genome_sample
