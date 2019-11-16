import math
import multiprocessing
from multiprocessing.pool import Pool
from time import time

import torch
from torch.utils.data import DataLoader, Dataset

from experiments.logger import logger
from neat.configuration import ConfigError, get_configuration
from neat.dataset.classification_example import ClassificationExample1Dataset
from neat.dataset.classification_mnist import MNISTReducedDataset
from neat.dataset.regression_example import RegressionExample1Dataset, RegressionExample2Dataset
from neat.fitness.kl_divergence import compute_kl_qw_pw
from neat.genome import Genome
from neat.loss.vi_loss import get_loss, get_beta
from neat.representation_mapping.genome_to_network.complex_stochastic_network import ComplexStochasticNetwork
from neat.representation_mapping.genome_to_network.stochastic_network import StochasticNetwork
from neat.utils import timeit


class CustomeDataLoader:
    def __init__(self, dataset: Dataset, shuffle=True):
        self.dataset = dataset
        self.shuffle = shuffle

        self.iterator = self._get_iterator()

    def _get_iterator(self):
        return self.dataset.x, self.dataset.y

    def __iter__(self):
        return self.iterator

    def __len__(self):
        return len(self.dataset)


class NoDaemonProcess(multiprocessing.Process):
    # make 'daemon' attribute always return False
    def _get_daemon(self):
        return False
    def _set_daemon(self, value):
        pass
    daemon = property(_get_daemon, _set_daemon)


class MyPool(multiprocessing.pool.Pool):
    Process = NoDaemonProcess


def get_data_loader(dataset: Dataset, batch_size=None):
    config = get_configuration()
    parallel_evaluation = config.parallel_evaluation
    if not parallel_evaluation:
        return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=1)
    else:
        return CustomeDataLoader(dataset, shuffle=True)


class EvaluationStochasticEngine:
    def __init__(self, testing=False, batch_size=None):
        self.config = get_configuration()
        self.batch_size = batch_size if batch_size is not None else self.config.batch_size
        self.parallel_evaluation = self.config.parallel_evaluation
        self.is_gpu = self.config.is_gpu
        self.dataset = get_dataset(self.config.dataset_name, testing=testing)
        self.dataset.generate_data()

        self.data_loader = get_data_loader(dataset=self.dataset, batch_size=self.batch_size)
        self.m = math.ceil(len(self.data_loader) / self.batch_size)

        self.loss = get_loss(problem_type=self.config.problem_type)

    @timeit
    def evaluate(self, population: dict):
        # TODO: make n_samples increase with generation number
        n_samples = self.config.n_samples
        if self.parallel_evaluation:
            n_cpus = multiprocessing.cpu_count()
            pool = MyPool(min(n_cpus//2, 8))
            tasks = []
            for genome in population.values():
                logger.debug(f'Genome {genome.key}: {genome.get_graph()}')
                x = (genome, self.dataset, self.loss, self.config.beta_type, self.config.problem_type,
                     self.batch_size, n_samples, self.is_gpu)
                tasks.append(x)

            # TODO: fix logging when using multiprocessing. Easy fix is to disable
            fitnesses = list(pool.imap(evaluate_genome_parallel, tasks, chunksize=len(population)//n_cpus))

            pool.close()
            for i, genome in enumerate(population.values()):
                genome.fitness = fitnesses[i]

        else:
            for genome in population.values():
                logger.debug(f'Genome {genome.key}: {genome.get_graph()}')
                genome.fitness = - evaluate_genome(genome=genome,
                                                   problem_type=self.config.problem_type,
                                                   data_loader=self.data_loader,
                                                   loss=self.loss,
                                                   beta_type=self.config.beta_type,
                                                   batch_size=self.batch_size,
                                                   n_samples=n_samples,
                                                   is_gpu=self.is_gpu)

        return population


def evaluate_genome_parallel(x):
    return - evaluate_genome2(*x)


def evaluate_genome2(genome: Genome, dataset, loss, beta_type, problem_type,
                     batch_size=10000, n_samples=10, is_gpu=False):
    '''
    Calculates: KL-Div(q(w)||p(w|D))
    Uses the VariationalInferenceLoss class (not the alternative)
    '''
    kl_posterior = 0

    kl_qw_pw = compute_kl_qw_pw(genome=genome)

    # setup network
    network = ComplexStochasticNetwork(genome=genome)
    if is_gpu:
        network.cuda()

    m = math.ceil(len(dataset) / batch_size)

    network.eval()

    # calculate Data log-likelihood (p(y*|x*,D))
    x_batch, y_batch = dataset.x, dataset.y
    x_batch, y_batch = _prepare_batch_data(x_batch=x_batch,
                                           y_batch=y_batch,
                                           problem_type=problem_type,
                                           is_gpu=is_gpu,
                                           n_input=genome.n_input,
                                           n_output=genome.n_output,
                                           n_samples=n_samples)

    with torch.no_grad():
        # forward pass
        output, _ = network(x_batch)
        # print(self.config.beta_type)
        beta = get_beta(beta_type=beta_type, m=m, batch_idx=0, epoch=1, n_epochs=1)
        # print(f'Beta: {beta}')
        kl_posterior += loss(y_pred=output, y_true=y_batch, kl_qw_pw=kl_qw_pw, beta=beta)

    loss_value = kl_posterior.item()
    return loss_value


@timeit
def evaluate_genome(genome: Genome, data_loader, loss, beta_type, problem_type,
                    batch_size=10000, n_samples=10, is_gpu=False, return_all=False):
    '''
    Calculates: KL-Div(q(w)||p(w|D))
    Uses the VariationalInferenceLoss class (not the alternative)
    '''
    kl_posterior = 0

    kl_qw_pw = compute_kl_qw_pw(genome=genome)

    # setup network
    network = ComplexStochasticNetwork(genome=genome)
    if is_gpu:
        network.cuda()

    m = math.ceil(len(data_loader) / batch_size)

    network.eval()

    chunks_x = []
    chunks_y_pred = []
    chunks_y_true = []

    # calculate Data log-likelihood (p(y*|x*,D))
    for batch_idx, (x_batch, y_batch) in enumerate(data_loader):

        x_batch, y_batch = _prepare_batch_data(x_batch=x_batch,
                                               y_batch=y_batch,
                                               problem_type=problem_type,
                                               is_gpu=is_gpu,
                                               n_input=genome.n_input,
                                               n_output=genome.n_output,
                                               n_samples=n_samples)

        with torch.no_grad():
            # forward pass
            output, _ = network(x_batch)

            beta = get_beta(beta_type=beta_type, m=m, batch_idx=batch_idx, epoch=1, n_epochs=1)

            kl_posterior += loss(y_pred=output, y_true=y_batch, kl_qw_pw=kl_qw_pw, beta=beta)
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


@timeit
def _prepare_batch_data(x_batch, y_batch, is_gpu, n_input, n_output, problem_type, n_samples):
    x_batch = x_batch.view(-1, n_input).repeat(n_samples, 1)
    if problem_type == 'classification':
        y_batch = y_batch.view(-1, 1).repeat(n_samples, 1).squeeze()
    elif problem_type == 'regression':
        y_batch = y_batch.view(-1, n_output).repeat(n_samples, 1)
    else:
        raise ValueError(f'Problem Type is not correct: {problem_type}')

    if is_gpu:
        x_batch, y_batch = x_batch.cuda(), y_batch.cuda()

    return x_batch, y_batch


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
    elif dataset_name == 'classification_mnist_reduced':
        dataset = MNISTReducedDataset(dataset_type=dataset_type)
        dataset.generate_data()
        return dataset
    else:
        raise ConfigError(f'Dataset Name is incorrect: {dataset_name}')
