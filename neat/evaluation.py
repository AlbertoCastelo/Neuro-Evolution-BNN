import math
import torch
from torch.utils.data import DataLoader

from neat.configuration import ConfigError, get_configuration
from neat.dataset.classification_example import ClassificationExample1Dataset
from neat.dataset.regression_example import RegressionExample1Dataset, RegressionExample2Dataset
from neat.fitness.kl_divergence import compute_kl_qw_pw, compute_kl_qw_pw_by_sum
from neat.genome import Genome
from neat.loss.vi_loss import get_loss_alternative, get_loss
from neat.representation_mapping.genome_to_network.deterministic_network import DeterministicNetwork
from neat.representation_mapping.genome_to_network.stochastic_network import StochasticNetworkOld, StochasticNetwork


class EvaluationEngine:
    def __init__(self, testing=False, batch_size=None):
        self.config = get_configuration()
        self.batch_size = batch_size if batch_size is not None else self.config.batch_size

        self.dataset = get_dataset(self.config.dataset_name, testing=testing)
        self.data_loader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True, num_workers=4)
        self.m = math.ceil(len(self.data_loader) / self.batch_size)

        self.loss = get_loss(problem_type=self.config.problem_type)

    def evaluate(self, population: dict):
        # TODO: make n_samples increase with generation number
        n_samples = self.config.n_samples

        for key, genome in population.items():
            genome.fitness = self.evaluate_genome(genome=genome,
                                                  n_samples=n_samples)

        return population

    def evaluate_genome(self, genome: Genome, n_samples=10, is_gpu=False, return_all=False):
        '''
        Calculates: KL-Div(q(w)||p(w|D))
        Uses the VariationalInferenceLoss class (not the alternative)
        '''
        kl_posterior = 0

        kl_qw_pw = compute_kl_qw_pw(genome=genome)

        # setup network
        network = StochasticNetworkOld(genome=genome)
        if is_gpu:
            network.cuda()
        network.eval()

        # calculate Data log-likelihood (p(y*|x*,D))
        chunks_x = []
        chunks_y_pred = []
        chunks_y_true = []
        for i in range(n_samples):
            for x_batch, y_batch in self.data_loader:
                x_batch = x_batch.reshape((-1, genome.n_input))
                x_batch = x_batch.float()

                y_batch = y_batch.float()
                if is_gpu:
                    x_batch, y_batch = x_batch.cuda(), y_batch.cuda()

                with torch.no_grad():
                    # forward pass
                    output = network(x_batch)
                    chunks_x.append(x_batch)
                    chunks_y_pred.append(output)
                    chunks_y_true.append(y_batch)
        x = torch.cat(chunks_x, dim=0)
        y_pred = torch.cat(chunks_y_pred, dim=0)
        y_true = torch.cat(chunks_y_true, dim=0)
        kl_posterior = self.loss(y_pred=y_pred, y_true=y_true, kl_qw_pw=kl_qw_pw, beta=self.get_beta())
        if return_all:
            return x, y_true, y_pred, kl_posterior.item(), kl_qw_pw
        return kl_posterior.item()

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


class EvaluationStochasticGoodEngine:
    def __init__(self, testing=False, batch_size=None):
        self.config = get_configuration()
        self.batch_size = batch_size if batch_size is not None else self.config.batch_size

        self.dataset = get_dataset(self.config.dataset_name, testing=testing)
        self.dataset.generate_data()
        self.data_loader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True, num_workers=4)
        self.m = math.ceil(len(self.data_loader) / self.batch_size)

        self.loss = get_loss(problem_type=self.config.problem_type)

    def evaluate(self, population: dict):
        # TODO: make n_samples increase with generation number
        n_samples = self.config.n_samples

        for key, genome in population.items():
            genome.fitness = self.evaluate_genome(genome=genome,
                                                  n_samples=n_samples)

        return population

    def evaluate_genome(self, genome: Genome, n_samples=10, is_gpu=False, return_all=False):
        '''
        Calculates: KL-Div(q(w)||p(w|D))
        Uses the VariationalInferenceLoss class (not the alternative)
        '''
        kl_posterior = 0

        kl_qw_pw = compute_kl_qw_pw(genome=genome)
        # print(kl_qw_pw)
        # print(f'kl_prior by sum: {compute_kl_qw_pw_by_sum(genome)}')

        # setup network
        network = StochasticNetwork(genome=genome, n_samples=n_samples)
        if is_gpu:
            network.cuda()
        network.eval()

        chunks_x = []
        chunks_y_pred = []
        chunks_y_true = []

        # calculate Data log-likelihood (p(y*|x*,D))
        for x_batch, y_batch in self.data_loader:
            x_batch = x_batch.reshape((-1, genome.n_input))
            x_batch = x_batch.view(-1, genome.n_input).repeat(n_samples, 1)

            y_batch = y_batch.view(-1, 1).repeat(n_samples, 1).squeeze()
            if is_gpu:
                x_batch, y_batch = x_batch.cuda(), y_batch.cuda()

            with torch.no_grad():
                # forward pass
                output, _ = network(x_batch)

                kl_posterior += self.loss(y_pred=output, y_true=y_batch, kl_qw_pw=kl_qw_pw, beta=self.get_beta())
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


class EvaluationAlternativeEngine:
    def __init__(self):
        self.config = get_configuration()
        self.batch_size = self.config.batch_size

        self.dataset = get_dataset(self.config.dataset_name)
        self.data_loader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True, num_workers=4)

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
                # TODO: be careful with this reshape when dimensions increase!!!!!
                x_batch = x_batch.reshape((-1, genome.n_input))
                x_batch = x_batch.float()
                y_batch = y_batch.float()
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
