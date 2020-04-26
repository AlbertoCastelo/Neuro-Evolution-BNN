import math
from multiprocessing import Manager, Queue

import torch

from experiments.multiprocessing_utils import Worker
from neat.evaluation.evaluate_simple import _process_output_data
from neat.evaluation.utils import get_dataset, _prepare_batch_data
from neat.fitness.kl_divergence import compute_kl_qw_pw
from neat.genome import Genome
from neat.loss.vi_loss import get_beta, get_loss
from neat.representation_mapping.genome_to_network.complex_stochastic_network import ComplexStochasticNetwork
from neat.representation_mapping.genome_to_network.complex_stochastic_network_jupyneat import \
    ComplexStochasticNetworkJupyneat
from neat.utils import timeit


def process_initialization(dataset_name, train_percentage, testing, dataset_random_state, noise):
    global dataset
    dataset = get_dataset(dataset_name, train_percentage=train_percentage, testing=testing,
                          random_state=dataset_random_state, noise=noise)


def evaluate_genome_task(x):
    return - _evaluate_genome_parallel(*x)


def evaluate_genome_task_jupyneat(x):
    return - _evaluate_genome_parallel_jupyneat(*x)


@timeit
def _evaluate_genome_parallel(genome: Genome, loss, beta_type, problem_type, is_testing,
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
    if is_testing:
        x_batch, y_batch = dataset.x_test, dataset.y_test
    else:
        x_batch, y_batch = dataset.x_train, dataset.y_train
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
        # output, _, y_batch = _process_output_data(output, y_true=y_batch, n_samples=n_samples,
        #                                           n_output=genome.n_output, problem_type=problem_type, is_pass=True)
        beta = get_beta(beta_type=beta_type, m=m, batch_idx=0, epoch=1, n_epochs=1)
        kl_posterior += loss(y_pred=output, y_true=y_batch, kl_qw_pw=kl_qw_pw, beta=beta)

    loss_value = kl_posterior.item()
    return loss_value


def _evaluate_genome_parallel_jupyneat(genome: dict, loss, beta_type, problem_type, n_input, n_output, activation,
                                       batch_size=10000, n_samples=10, is_gpu=False):
    '''
    Calculates: KL-Div(q(w)||p(w|D))
    Uses the VariationalInferenceLoss class (not the alternative)
    '''

    kl_posterior = 0
    # TODO: fix
    # kl_qw_pw = compute_kl_qw_pw(genome=genome)
    kl_qw_pw = 0.0

    # setup network
    network = ComplexStochasticNetworkJupyneat(genome=genome, n_input=n_input, n_output=n_output,
                                               activation_type=activation)
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
                                           n_input=n_input,
                                           n_output=n_output,
                                           n_samples=n_samples)

    if is_gpu:
        x_batch, y_batch = x_batch.cuda(), y_batch.cuda()

    with torch.no_grad():
        # forward pass
        output, _ = network(x_batch)
        beta = get_beta(beta_type=beta_type, m=m, batch_idx=0, epoch=1, n_epochs=1)
        kl_posterior += loss(y_pred=output, y_true=y_batch, kl_qw_pw=kl_qw_pw, beta=beta)

    loss_value = kl_posterior.item()
    return loss_value


def _parallelize_with_workers(n_cpus, population):
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


# TODO: this could be removed as it is not used
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
