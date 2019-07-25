from unittest import TestCase, skip

from neat.genome import GenomeSample
from neat.representation.deterministic_network import DeterministicNetwork
import torch


class TestRepresentationFeedForwardWithoutHiddenLayers(TestCase):

    def test_network_structure(self):
        node_genes = {0: 1.0,
                      1: 0.0}

        connection_genes = {(-1, 0): 1.5,
                            (-2, 0): 2.5,
                            (-1, 1): -0.5,
                            (-2, 1): 0.0}
        n_output = 2
        genome_sample = GenomeSample(key=None, n_input=2, n_output=n_output,
                                     node_genes=node_genes,
                                     connection_genes=connection_genes)

        model = DeterministicNetwork(genome=genome_sample)

        input = torch.tensor([1.0, -1.0])
        result = model(input.data)

        self.assertEqual(len(result), n_output)

    def test_network_structure_with_batch_size(self):
        node_genes = {0: 1.0,
                      1: 0.0}

        connection_genes = {(-1, 0): 1.5,
                            (-2, 0): 2.5,
                            (-1, 1): -0.5,
                            (-2, 1): 0.0}
        n_output = 2
        genome_sample = GenomeSample(key=None, n_input=2, n_output=n_output,
                                     node_genes=node_genes,
                                     connection_genes=connection_genes)

        model = DeterministicNetwork(genome=genome_sample)

        input = torch.tensor([[1.0, -1.0],
                              [0.5, 0],
                              [0, 2.5]])
        result = model(input.data)

        self.assertEqual(result.shape[0], 3)
        self.assertEqual(result.shape[1], n_output)

    def test_connection_is_not_specified_assumes_0(self):
        node_genes = {0: 1.0,
                      1: 0.0}

        connection_genes = {(-1, 0): 1.5,
                            (-2, 0): 2.5,
                            (-1, 1): -0.5}
        n_output = 2
        genome_sample = GenomeSample(key=None, n_input=2, n_output=n_output,
                                     node_genes=node_genes,
                                     connection_genes=connection_genes)

        model = DeterministicNetwork(genome=genome_sample)

        input = torch.tensor([1.0, -1.0])
        result = model(input.data)
        missing_weight = model.state_dict()['layer_0.weight'].data[1, 1]
        self.assertEqual(0.0, missing_weight)
        self.assertEqual(len(result), n_output)

    def test_weights_are_correctly_set(self):
        node_genes = {0: 0.0,
                      1: 0.0}

        connection_genes = {(-1, 0): 1.5,
                            (-2, 0): 2.5,
                            (-1, 1): 0.0,
                            (-2, 1): 0.0}
        n_output = 2
        genome_sample = GenomeSample(key=None, n_input=2, n_output=n_output,
                                     node_genes=node_genes,
                                     connection_genes=connection_genes)

        # model = get_nn_from_genome(genome=genome_sample)
        model = DeterministicNetwork(genome=genome_sample)
        input = torch.tensor([1.0, 1.0])
        result = model(input.data)

        self.assertEqual(len(result), n_output)

        self.assertTrue(torch.allclose(result, torch.tensor([4.0, 0.0]), atol=1e-02))

    def test_bias_are_correctly_set(self):
        node_genes = {0: 1.0,
                      1: 0.0}

        connection_genes = {(-1, 0): 0.0,
                            (-2, 0): 0.0,
                            (-1, 1): 0.0,
                            (-2, 1): 0.0}
        n_output = 2
        genome_sample = GenomeSample(key=None, n_input=2, n_output=n_output,
                                     node_genes=node_genes,
                                     connection_genes=connection_genes)

        # model = get_nn_from_genome(genome=genome_sample)
        model = DeterministicNetwork(genome=genome_sample)
        input = torch.tensor([1.0, 1.0])
        result = model(input.data)

        self.assertEqual(len(result), n_output)
        self.assertTrue(torch.allclose(result, torch.tensor([1.0, 0.0]), atol=1e-03))


class TestRepresentationFeedForwardWithOneHiddenLayers(TestCase):

    def test_network_structure(self):
        genome_sample = generate_feedforward_with_one_hidden_unit()

        model = DeterministicNetwork(genome=genome_sample)

        input = torch.tensor([1.0, -1.0])
        result = model(input.data)

        self.assertEqual(len(result), genome_sample.n_output)
        self.assertEqual(model.n_layers, 2)
        self.assertTrue(torch.allclose(result, torch.tensor([2.4065, 1.0]), atol=1e-02))

    def test_weights_are_located_correctly(self):
        genome_sample = generate_feedforward_with_one_hidden_unit()

        model = DeterministicNetwork(genome=genome_sample)

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
    node_genes = {0: 1.0,
                  1: 1.0,
                  2: 1.0,
                  3: -1.0,
                  4: 0}

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


class TestRepresentationFeedForwardWithJumpingConnections(TestCase):

    @skip('Not there yet')
    def test_network_structure(self):
        genome_sample = generate_feedforward_with_jumping_connections()

        model = DeterministicNetwork(genome=genome_sample)

        input = torch.tensor([1.0, -1.0])
        result = model.forward(input.data)

        self.assertEqual(len(result), genome_sample.n_output)
        self.assertEqual(model.n_layers, 2)
        self.assertTrue(torch.allclose(result, torch.tensor([0.9173, 0.7311]), atol=1e-02))


def generate_feedforward_with_jumping_connections():
    pass