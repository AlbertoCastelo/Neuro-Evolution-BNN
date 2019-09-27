import math
import torch
from torch.utils.data import DataLoader

from neat.configuration import ConfigError, get_configuration
from neat.dataset.classification_example import ClassificationExample1Dataset
from neat.dataset.regression_example import RegressionExample1Dataset, RegressionExample2Dataset
from neat.fitness.kl_divergence import compute_kl_qw_pw
from neat.genome import Genome
from neat.loss.vi_loss import get_loss, get_beta
from neat.representation_mapping.genome_to_network.stochastic_network import StochasticNetwork
from neat.utils import timeit


class EvaluationStochasticEngine:
    def __init__(self, testing=False, batch_size=None):
        self.config = get_configuration()
        self.batch_size = batch_size if batch_size is not None else self.config.batch_size

        self.dataset = get_dataset(self.config.dataset_name, testing=testing)
        self.dataset.generate_data()
        self.data_loader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True, num_workers=4)
        self.m = math.ceil(len(self.data_loader) / self.batch_size)

        self.loss = get_loss(problem_type=self.config.problem_type)

    @timeit
    def evaluate(self, population: dict):
        # TODO: make n_samples increase with generation number
        n_samples = self.config.n_samples

        for key, genome in population.items():
            genome.fitness = - self.evaluate_genome(genome=genome,
                                                    n_samples=n_samples)

        return population

    def evaluate_genome(self, genome: Genome, n_samples=10, is_gpu=False, return_all=False):
        '''
        Calculates: KL-Div(q(w)||p(w|D))
        Uses the VariationalInferenceLoss class (not the alternative)
        '''
        kl_posterior = 0

        kl_qw_pw = compute_kl_qw_pw(genome=genome)
        # print(f'KL-prior: {kl_qw_pw}')
        # print(f'kl_prior by sum: {compute_kl_qw_pw_by_sum(genome)}')

        # setup network
        network = StochasticNetwork(genome=genome, n_samples=n_samples)
        if is_gpu:
            network.cuda()

        m = math.ceil(len(self.data_loader) / self.batch_size)

        network.eval()

        chunks_x = []
        chunks_y_pred = []
        chunks_y_true = []
        # calculate Data log-likelihood (p(y*|x*,D))
        for batch_idx, (x_batch, y_batch) in enumerate(self.data_loader):
            x_batch = x_batch.view(-1, genome.n_input).repeat(n_samples, 1)

            y_batch = y_batch.view(-1, 1).repeat(n_samples, 1).squeeze()
            if is_gpu:
                x_batch, y_batch = x_batch.cuda(), y_batch.cuda()

            with torch.no_grad():
                # forward pass
                output, _ = network(x_batch)
                # print(self.config.beta_type)
                beta = get_beta(beta_type=self.config.beta_type, m=m, batch_idx=batch_idx, epoch=1, n_epochs=1)
                # print(f'Beta: {beta}')
                kl_posterior += self.loss(y_pred=output, y_true=y_batch, kl_qw_pw=kl_qw_pw, beta=beta)
                if return_all:
                    chunks_x.append(x_batch)
                    chunks_y_pred.append(output)
                    chunks_y_true.append(y_batch)

        loss_value = kl_posterior.item()

        if return_all:
            x = torch.cat(chunks_x, dim=0)
            y_pred = torch.cat(chunks_y_pred, dim=0)
            y_true = torch.cat(chunks_y_true, dim=0)
            return x, y_true, y_pred, loss_value
        return loss_value


def get_dataset(dataset_name, testing=False):
    if testing:
        dataset_type = 'test'
    else:
        dataset_type = 'train'

    if dataset_name == 'regression_example_1':
        dataset = RegressionExample1Dataset(dataset_type=dataset_type)
        dataset.generate_data()
        return dataset
    elif dataset_name == 'regression_example_2':
        dataset = RegressionExample2Dataset(dataset_type=dataset_type)
        dataset.generate_data()
        return dataset
    elif dataset_name == 'classification_example_1':
        dataset = ClassificationExample1Dataset(dataset_type=dataset_type)
        dataset.generate_data()
        return dataset
    else:
        raise ConfigError(f'Dataset Name is incorrect: {dataset_name}')
