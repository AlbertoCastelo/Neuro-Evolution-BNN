import torch
from torch.distributions import kl_divergence, Distribution, MultivariateNormal, Normal

from neat.configuration import get_configuration
from neat.genome import Genome
import numpy as np

def compute_kl_qw_pw(genome: Genome):
    '''
    https://pytorch.org/docs/stable/distributions.html#module-torch.distributions.kl
    '''
    pw = get_pw(genome)

    qw = get_qw(genome)

    kl_qw_pw = kl_divergence(qw, pw).sum()
    return kl_qw_pw


def get_pw(genome: Genome) -> Distribution:

    # get prior configuration
    config = genome.genome_config
    means = []
    stds = []
    for key, node in genome.node_genes.items():
        means.append(config.bias_mean_prior)
        stds.append(config.bias_std_prior)

    for key, connection in genome.connection_genes.items():
        means.append(config.weight_mean_prior)
        stds.append(config.weight_std_prior)

    mu = torch.Tensor(means)
    sigma = torch.Tensor(stds)
    return Normal(loc=mu, scale=sigma)


def get_qw(genome: Genome) -> Distribution:
    means = []
    stds = []
    for key, node in genome.node_genes.items():
        means.append(node.get_mean())
        stds.append(node.get_std())

    for key, connection in genome.connection_genes.items():
        means.append(connection.get_mean())
        stds.append(connection.get_std())

    mu = torch.Tensor(means)
    sigma = torch.Tensor(stds)
    return Normal(loc=mu, scale=sigma)


# TODO: this equation is wrong!!
def compute_kl_qw_pw_by_sum(genome: Genome):
    # get prior configuration
    config = get_configuration()
    kl_qw_pw = 0.0

    for key, node in genome.node_genes.items():
        pb = Normal(loc=config.bias_mean_prior, scale=config.bias_std_prior)
        qb = Normal(loc=node.get_mean(), scale=np.exp(1.0 + node.get_std()))
        kl_qw_pw += kl_divergence(qb, pb)

    for key, connection in genome.connection_genes.items():
        pw = Normal(loc=config.weight_mean_prior, scale=config.weight_std_prior)
        qw = Normal(loc=connection.get_mean(), scale=np.exp(1.0 + connection.get_std()))
        kl_qw_pw += kl_divergence(qw, pw)
    return kl_qw_pw