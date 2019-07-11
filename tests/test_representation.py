from unittest import TestCase

from neat.gene import NodeGene
from neat.genome import GenomeSample
from neat.representation import get_nn_from_genome, Network
import torch
import numpy as np


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

    def test_weights_are_correctly_set(self):
        node_0 = NodeGene(key=0)
        node_0.random_initialization()
        node_1 = NodeGene(key=1)
        node_1.random_initialization()

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
