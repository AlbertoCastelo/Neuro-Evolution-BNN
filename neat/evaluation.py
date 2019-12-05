import math
from multiprocessing import cpu_count, Queue, Manager, Pool

import torch
from torch.utils.data import DataLoader, Dataset

from experiments.logger import logger
from experiments.multiprocessing_utils import Worker
from neat.configuration import ConfigError, get_configuration
from neat.dataset.classification_example import ClassificationExample1Dataset
from neat.dataset.classification_mnist import MNISTDataset
from neat.dataset.classification_mnist_binary import MNISTBinaryDataset
from neat.dataset.custom_dataloader import get_data_loader
from neat.dataset.regression_example import RegressionExample1Dataset, RegressionExample2Dataset
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
            self.n_processes = min(cpu_count() // 2, 8)
            self.pool = Pool(processes=self.n_processes, initializer=process_initialization, initargs=(self.config.dataset_name, True))

    @timeit
    def evaluate(self, population: dict):
        # TODO: make n_samples increase with generation number
        n_samples = self.config.n_samples
        if self.parallel_evaluation:
            tasks = []

            for genome in population.values():
                logger.debug(f'Genome {genome.key}: {genome.get_graph()}')
                x = (genome.copy(), get_loss(problem_type=self.config.problem_type),
                     self.config.beta_type, self.config.problem_type,
                     self.batch_size, n_samples, self.is_gpu)
                tasks.append(x)

            # TODO: fix logging when using multiprocessing. Easy fix is to disable
            fitnesses = list(self.pool.imap(evaluate_genome_task, tasks, chunksize=len(population)//self.n_processes))

            for i, genome in enumerate(population.values()):
                genome.fitness = fitnesses[i]

        else:
            self.dataset = self._get_dataset()
            self.data_loader = self._get_dataloader()
            self.loss = self._get_loss()
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

    def close(self):
        if self.parallel_evaluation:
            self.pool.close()

    def _parallelize_with_workers(self, n_cpus, population):
        # create workers
        manager = Manager()
        task_queue = manager.Queue()
        exit_queue = manager.Queue()
        exception_queue = manager.Queue()
        results_queue = manager.Queue()
        # task_queue = SimpleQueue()
        # exit_queue = SimpleQueue()
        # exception_queue = SimpleQueue()
        # results_queue = SimpleQueue()
        workers = []
        for i in range(n_cpus):
            worker = Worker(task_queue=task_queue,
                            exit_queue=exit_queue,
                            exception_queue=exception_queue,
                            results_queue=results_queue)
            worker.start()
            workers.append(worker)
        for genome in population.values():
            task_queue.put(Task(genome=genome.copy(), dataset=None,
                                x=torch.zeros((2, 784)).float(),
                                y=torch.zeros(2).short(),
                                # x=self.dataset.x.clone().detach(),
                                # y=self.dataset.y.clone().detach(),
                                loss=get_loss('classification'),
                                beta_type='other',
                                problem_type='classification',
                                batch_size=100000,
                                n_samples=20,
                                is_gpu=False))
            # loss=self.loss,
            # beta_type=self.config.beta_type, problem_type=self.config.problem_type,
            # batch_size=self.batch_size, n_samples=n_samples, is_gpu=self.is_gpu))
        while not task_queue.empty():
            print('reading results from workers')
            print(task_queue.qsize())
            # for worker in workers:
            #     print(f'Is alive: {worker.is_alive()}')
            # print('reading results from workers')
            # print(task_queue.qsize())
            if not exception_queue.empty():
                exception = exception_queue.get()
                raise exception
            results = results_queue.get()
            print(results)
            population[results[0]].fitness = - results[1]
        # sys.exit()
        # terminate workers
        # TODO: workers can live during the whole process
        for i in range(n_cpus):
            print('sending exit conditions')
            exit_queue.put(1)

    def _get_dataset(self):
        if self.dataset is None:
            self.dataset = get_dataset(self.config.dataset_name, testing=self.testing)
            self.dataset.generate_data()
        return self.dataset

    def _get_dataloader(self):
        if self.data_loader is None:
            self.data_loader = get_data_loader(dataset=self.dataset, batch_size=self.batch_size)
            # self.m = math.ceil(len(self.data_loader) / self.batch_size)
        return self.data_loader

    def _get_loss(self):
        if self.loss is None:
            self.loss = get_loss(problem_type=self.config.problem_type)
        return self.loss


def process_initialization(dataset_name, testing):
    global dataset
    dataset = get_dataset(dataset_name, testing=testing)
    dataset.generate_data()


def evaluate_genome_task(x):
    return - _evaluate_genome_parallel(*x)


class Task:
    def __init__(self, genome: Genome, dataset, x, y, loss, beta_type, problem_type,
                 # config,
                 batch_size=10000, n_samples=10, is_gpu=False, queue: Queue = None):
        # super().__init__()
        self.queue = queue
        self.genome = genome
        self.dataset = dataset
        self.x = x
        self.y = y
        self.loss = loss
        self.beta_type = beta_type
        self.problem_type = problem_type
        self.batch_size = batch_size
        self.n_samples = n_samples
        self.is_gpu = False
        self.config = genome.genome_config

        self.result = None

    def __str__(self):
        return f'Genome: {self.genome.key}'

    def run(self) -> None:
        '''
            Calculates: KL-Div(q(w)||p(w|D))
            Uses the VariationalInferenceLoss class (not the alternative)
            '''
        # from experiments.multiprocessing_utils import ForkedPdb; ForkedPdb().set_trace()
        kl_posterior = 0

        kl_qw_pw = compute_kl_qw_pw(genome=self.genome)

        # setup network
        network = ComplexStochasticNetwork(genome=self.genome)

        # m = math.ceil(len(self.dataset) / self.batch_size)
        m = math.ceil(len(self.x) / self.batch_size)

        network.eval()

        # calculate Data log-likelihood (p(y*|x*,D))
        # x_batch, y_batch = self.dataset.x, self.dataset.y
        x_batch, y_batch = self.x, self.y
        x_batch, y_batch = _prepare_batch_data(x_batch=x_batch,
                                               y_batch=y_batch,
                                               problem_type=self.problem_type,
                                               is_gpu=self.is_gpu,
                                               n_input=self.genome.n_input,
                                               n_output=self.genome.n_output,
                                               n_samples=self.n_samples)
        print('running forward pass')
        with torch.no_grad():
            # forward pass
            output, _ = network(x_batch)
            print('forward pass completed')
            # print(self.config.beta_type)
            beta = get_beta(beta_type=self.beta_type, m=m, batch_idx=0, epoch=1, n_epochs=1)
            # print(f'Beta: {beta}')
            kl_posterior += self.loss(y_pred=output, y_true=y_batch, kl_qw_pw=kl_qw_pw, beta=beta)

        loss_value = kl_posterior.item()
        self.result = (self.genome.key, loss_value)
        # self.queue.put()

    def get_results(self):
        return self.result


def _evaluate_genome_parallel(genome: Genome, loss, beta_type, problem_type,
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
    m = math.ceil(len(dataset.x) / batch_size)

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

    if is_gpu:
        x_batch, y_batch = x_batch.cuda(), y_batch.cuda()

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
            print(f'output-shape: {output.shape}')
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
    elif dataset_name == 'regression_example_2':
        dataset = RegressionExample2Dataset(dataset_type=dataset_type)
    elif dataset_name == 'classification_example_1':
        dataset = ClassificationExample1Dataset(dataset_type=dataset_type)
    elif dataset_name == 'mnist':
        dataset = MNISTDataset(dataset_type=dataset_type)
    elif dataset_name == 'mnist_binary':
        dataset = MNISTBinaryDataset(dataset_type=dataset_type)
    else:
        raise ConfigError(f'Dataset Name is incorrect: {dataset_name}')
    dataset.generate_data()
    return dataset
