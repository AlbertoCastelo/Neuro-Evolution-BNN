from unittest import TestCase

import torch
from torch.distributions import MultivariateNormal, kl_divergence, Normal

from neat.configuration import get_configuration
from neat.fitness.kl_divergence import get_qw, get_pw, compute_kl_qw_pw, compute_kl_qw_pw_by_sum
from tests.config_files.config_files import get_config_files_path
from tests.utils.generate_genome import generate_genome_with_hidden_units


class TestPriorKLDivergence(TestCase):
    '''
    testing calculating KL(q(w)||p(w))
    '''
    def setUp(self) -> None:
        path = get_config_files_path()
        filename = ''.join([path, '/regression-miso.json'])
        config = get_configuration(filename=filename)
        self.genome = generate_genome_with_hidden_units(n_input=config.n_input,
                                                        n_output=config.n_output)

    def test_get_qw_from_genome(self):
        qw = get_qw(genome=self.genome)
        self.assertEqual(type(qw), Normal)

    def test_get_pw_from_genome(self):
        pw = get_pw(genome=self.genome)
        self.assertEqual(type(pw), Normal)

    def test_kl_from_genome(self):
        kl_qw_pw = compute_kl_qw_pw(genome=self.genome)

        self.assertEqual(type(kl_qw_pw.item()), float)

    def test_both_methods_give_same_result(self):
        kl_qw_pw = compute_kl_qw_pw(genome=self.genome)
        kl_qw_pw_2 = compute_kl_qw_pw_by_sum(self.genome)

        self.assertEqual(kl_qw_pw, kl_qw_pw_2)


class TestKLSameDistribution(TestCase):
    def test_kl_when_same_distribution(self):
        mean = torch.zeros(2)
        cov_matrix = torch.eye(2)
        p = MultivariateNormal(loc=mean, covariance_matrix=cov_matrix)

        kl = kl_divergence(p, p)

        self.assertEqual(kl.item(), 0.0)
