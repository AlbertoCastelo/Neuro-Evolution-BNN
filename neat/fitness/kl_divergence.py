import torch
from torch.distributions import kl_divergence, Distribution, MultivariateNormal, Normal

from neat.configuration import get_configuration
from neat.genome import Genome


def compute_kl_qw_pw(genome: Genome):
    '''
    https://pytorch.org/docs/stable/distributions.html#module-torch.distributions.kl
    '''
    pw = get_pw(genome)

    qw = get_qw(genome)

    kl_qw_pw = kl_divergence(qw, pw)
    # TODO: remove division by 100
    # return kl_qw_pw / 100
    return kl_qw_pw


def get_pw(genome: Genome) -> Distribution:
    # get prior configuration
    config = get_configuration()
    means = []
    stds = []
    for key, node in genome.node_genes.items():
        means.append(config.bias_mean_prior)
        stds.append(config.bias_std_prior)

    for key, connection in genome.connection_genes.items():
        means.append(config.weight_mean_prior)
        stds.append(config.weight_std_prior)

    mu = torch.Tensor(means)
    sigma_diag = torch.Tensor(stds)
    cov_matrix = torch.diag(sigma_diag)
    return MultivariateNormal(loc=mu, covariance_matrix=cov_matrix)


def get_qw(genome: Genome) -> Distribution:
    means = []
    stds = []
    for key, node in genome.node_genes.items():
        means.append(node.bias_mean)
        stds.append(node.bias_std)

    for key, connection in genome.connection_genes.items():
        means.append(connection.weight_mean)
        stds.append(connection.weight_std)

    mu = torch.Tensor(means)
    sigma_diag = torch.Tensor(stds)
    cov_matrix = torch.diag(sigma_diag)
    return MultivariateNormal(loc=mu, covariance_matrix=cov_matrix)


# TODO: this equation is wrong!!
def compute_kl_qw_pw_by_product(genome: Genome):
    # get prior configuration
    config = get_configuration()
    kl_qw_pw = torch.Tensor([1.0])

    for key, node in genome.node_genes.items():
        qw = Normal(loc=config.bias_mean_prior, scale=config.bias_std_prior)
        pw = Normal(loc=node.bias_mean, scale=node.bias_std)
        kl_qw_pw *= kl_divergence(qw, pw)

    for key, connection in genome.connection_genes.items():
        qw = Normal(loc=config.weight_mean_prior, scale=config.weight_std_prior)
        pw = Normal(loc=connection.weight_mean, scale=connection.weight_std)
        kl_qw_pw *= kl_divergence(qw, pw)
    return kl_qw_pw