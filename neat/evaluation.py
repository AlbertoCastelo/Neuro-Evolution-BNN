import math
import torch
from torch.utils.data import DataLoader

from neat.configuration import ConfigError, get_configuration
from neat.dataset.regression_example import RegressionExample1Dataset
from neat.fitness.kl_divergence import compute_kl_qw_pw
from neat.genome import Genome
from neat.loss.vi_loss import get_loss_alternative
from neat.representation import DeterministicNetwork, StochasticNetwork


class EvaluationEngine:
    def __init__(self):
        self.config = get_configuration()
        self.batch_size = self.config.batch_size

        self.dataset = get_dataset(self.config.dataset_name)
        self.data_loader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True, num_workers=4)
        self.m = math.ceil(len(self.data_loader) / self.batch_size)

        self.loss = get_loss_alternative(problem_type=self.config.problem_type)

    def evaluate(self, population: dict):
        # TODO: make n_samples increase with generation number
        n_samples = self.config.n_samples

        for key, genome in population.items():
            genome.fitness = self.evaluate_genome_alternative(genome=genome,
                                                              n_samples=n_samples)

        return population

    def evaluate_genome_alternative(self, genome: Genome, n_samples=10, is_gpu=False):
        '''
        Calculates: KL-Div(q(w)||p(w|D))
        This method of evaluating genome takes a sample of the network and calculates the log-likelihood.
        '''

        kl_qw_pw = compute_kl_qw_pw(genome=genome)
        log_py = 0
        for i in range(n_samples):
            genome_sample = genome.get_genome_sample()

            # setup network
            network = DeterministicNetwork(genome=genome_sample)
            if is_gpu:
                network.cuda()
            network.eval()

            # calculate Data log-likelihood (p(y*|x*,D))
            for batch_ids, (x_batch, y_batch) in enumerate(self.data_loader):
                x_batch = x_batch.reshape((-1, genome.n_input))
                if is_gpu:
                    x_batch, y_batch = x_batch.cuda(), y_batch.cuda()

                with torch.no_grad():
                    # forward pass
                    output = network(x_batch)
                    # log_py += self.loss(logits=output, y=y_batch, kl=kl_qw_pw, beta=self.get_beta())
                    log_py += self.loss(y_pred=output, y_true=y_batch)
        if is_gpu:
            log_py = log_py.cpu()
        kl_posterior = self.loss.compute_complete_loss(logpy=log_py, kl_qw_pw=kl_qw_pw)

        return kl_posterior.item()

    def evaluate_genome(self, genome: Genome, n_samples=10, is_gpu=False):
        '''
        Calculates: KL-Div(q(w)||p(w|D))
        Uses the VariationalInferenceLoss class (not the alternative)
        '''
        # TODO: each forward pass samples the network

        kl_posterior = 0

        kl_qw_pw = compute_kl_qw_pw(genome=genome)

        # setup network
        network = StochasticNetwork(genome=genome)
        if is_gpu:
            network.cuda()
        network.eval()

        # calculate Data log-likelihood (p(y*|x*,D))
        for x_batch, y_batch in self.data_loader:
            x_batch = x_batch.view(-1, 1).repeat(n_samples, 1)
            y_batch = y_batch.repeat(n_samples)
            if is_gpu:
                x_batch, y_batch = x_batch.cuda(), y_batch.cuda()

            with torch.no_grad():
                # forward pass
                output = network(x_batch)
                kl_posterior += self.loss(logits=output, y=y_batch, kl=kl_qw_pw, beta=self.get_beta())
        return kl_posterior

    def get_beta(self):

        # if self.beta_type is 'Blundell':
        #     beta = 2 ** (m - (batch_idx + 1)) / (2 ** m - 1)
        # elif self.beta_type is 'Soenderby':
        #     beta = min(epoch / (n_epochs // 4), 1)
        # elif self.beta_type is 'Standard':
        #     beta = 1 / m
        # else:
        #     beta = 0
        beta = 1 / self.m
        return beta


def get_dataset(dataset_name):
    if dataset_name == 'regression_example_1':
        return RegressionExample1Dataset()

    else:
        raise ConfigError(f'Dataset Name is incorrect: {dataset_name}')