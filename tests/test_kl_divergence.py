from unittest import TestCase

import torch
from torch import Tensor
from torch.distributions import MultivariateNormal, kl_divergence

from neat.fitness.kl_divergence import get_qw, get_pw, compute_kl_qw_pw
from neat.genome import Genome


class TestPriorKLDivergence(TestCase):
    '''
    testing calculating KL(q(w)||p(w))
    '''

    def test_kl_when_same_distribution(self):
        mean = torch.zeros(2)
        cov_matrix = torch.eye(2)
        p = MultivariateNormal(loc=mean, covariance_matrix=cov_matrix)

        kl = kl_divergence(p, p)

        self.assertEqual(kl.item(), 0.0)

    def test_get_qw_from_genome(self):
        genome = get_genome_with_hidden_layers()
        qw = get_qw(genome)
        self.assertEqual(type(qw), MultivariateNormal)

    def test_get_pw_from_genome(self):
        genome = get_genome_with_hidden_layers()
        pw = get_pw(genome)
        self.assertEqual(type(pw), MultivariateNormal)

    def test_kl_from_genome(self):
        genome = get_genome_with_hidden_layers()
        kl_qw_pw = compute_kl_qw_pw(genome)

        self.assertEqual(type(kl_qw_pw.item()), float)


def get_genome_with_hidden_layers():
    genome = Genome(key=1)
    genome.create_random_genome()

    return genome

