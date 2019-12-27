import math
from multiprocessing import cpu_count, Queue, Manager, Pool
import torch
from experiments.logger import logger
from experiments.multiprocessing_utils import Worker
from neat.configuration import ConfigError, get_configuration
from neat.dataset.classification_example import ClassificationExample1Dataset
from neat.dataset.classification_mnist import MNISTDataset
from neat.dataset.classification_mnist_binary import MNISTBinaryDataset
from neat.dataset.custom_dataloader import get_data_loader
from neat.dataset.regression_example import RegressionExample1Dataset, RegressionExample2Dataset
from neat.evaluation.evaluate_parallel import evaluate_genome_task, process_initialization
from neat.evaluation.evaluate_simple import evaluate_genome
from neat.evaluation.utils import get_dataset
from neat.fitness.kl_divergence import compute_kl_qw_pw
from neat.genome import Genome
from neat.loss.vi_loss import get_loss, get_beta
from neat.representation_mapping.genome_to_network.complex_stochastic_network import ComplexStochasticNetwork
from neat.utils import timeit


class EvaluationStochasticEngine:
    def __init__(self, testing=False, batch_size=None):
        self.config = get_configuration()
        self.testing = testing
        self.batch_size = batch_size if batch_size is not None else self.config.batch_size
        self.parallel_evaluation = self.config.parallel_evaluation
        self.is_gpu = self.config.is_gpu

        self.dataset = None
        self.data_loader = None
        self.loss = None

        if self.parallel_evaluation:
            self.n_processes = self._get_n_processes()
            self.pool = Pool(processes=self.n_processes, initializer=process_initialization, initargs=(self.config.dataset_name, True))

    def _get_n_processes(self):
        if self.config.n_processes is not None:
            return int(self.config.n_processes)
        return min(cpu_count() // 2, 8)


    @timeit
    def evaluate(self, population: dict):
        # TODO: make n_samples increase with generation number
        n_samples = self.config.n_samples
        if self.parallel_evaluation:
            tasks = []
            for genome in population.values():
                logger.debug(f'Genome {genome.key}: {genome.get_graph()}')
                x = (genome, get_loss(problem_type=self.config.problem_type),
                     self.config.beta_type, self.config.problem_type,
                     self.batch_size, n_samples, self.is_gpu)
                tasks.append(x)

            # TODO: fix logging when using multiprocessing. Easy fix is to disable
            fitnesses = list(self.pool.imap(evaluate_genome_task, tasks, chunksize=len(population)//self.n_processes))

            for i, genome in enumerate(population.values()):
                genome.fitness = fitnesses[i]

        else:
            self.dataset = self._get_dataset()
            # self.data_loader = self._get_dataloader()
            self.loss = self._get_loss()
            for genome in population.values():
                logger.debug(f'Genome {genome.key}: {genome.get_graph()}')
                genome.fitness = - evaluate_genome(genome=genome,
                                                   problem_type=self.config.problem_type,
                                                   dataset=self.dataset,
                                                   loss=self.loss,
                                                   beta_type=self.config.beta_type,
                                                   batch_size=self.batch_size,
                                                   n_samples=n_samples,
                                                   is_gpu=self.is_gpu)

        return population

    def close(self):
        if self.parallel_evaluation:
            self.pool.close()

    def _get_dataset(self):
        if self.dataset is None:
            self.dataset = get_dataset(self.config.dataset_name, testing=self.testing)
            self.dataset.generate_data()
        return self.dataset

    def _get_dataloader(self):
        if self.data_loader is None:
            self.data_loader = get_data_loader(dataset=self.dataset, batch_size=self.batch_size)
        return self.data_loader

    def _get_loss(self):
        if self.loss is None:
            self.loss = get_loss(problem_type=self.config.problem_type)
        return self.loss
