import math
import os
from multiprocessing.pool import Pool

import torch

from neat.evaluation import MyPool, _prepare_batch_data, get_dataset
from neat.fitness.kl_divergence import compute_kl_qw_pw
from neat.genome import Genome
from neat.loss.vi_loss import get_loss, get_beta
from neat.neat_logger import get_neat_logger
from neat.representation_mapping.genome_to_network.complex_stochastic_network import ComplexStochasticNetwork
from neat.utils import timeit
from tests.config_files.config_files import create_configuration

config = create_configuration(filename='/mnist_binary.json')
LOGS_PATH = f'{os.getcwd()}/'
logger = get_neat_logger(path=LOGS_PATH)

# N_SAMPLES = 10
N_PROCESSES = 16
N_GENOMES = 100

genomes = []
for i in range(N_GENOMES):
    genome = Genome(key=i)
    genome.create_random_genome()
    genomes.append(genome)


def evaluate_genome_parallel(x):
    return evaluate_genome(*x)


def process_initialization(dataset_name, testing):
    global dataset
    dataset = get_dataset(dataset_name, testing=testing)
    dataset.generate_data()

@timeit
def evaluate_genome(genome: Genome, loss, beta_type, problem_type,
                    batch_size=10000, n_samples=10, is_gpu=False):
    '''
    Calculates: KL-Div(q(w)||p(w|D))
    Uses the VariationalInferenceLoss class (not the alternative)
    # '''
    # dataset = get_dataset(genome.genome_config.dataset_name, testing=True)
    # dataset.generate_data()
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


tasks = []


pool = Pool(processes=N_PROCESSES, initializer=process_initialization, initargs=(config.dataset_name, True))
for genome in genomes:
    logger.debug(f'Genome {genome.key}: {genome.get_graph()}')

    # x = torch.zeros(2, 784).float()
    # y = torch.zeros(2).long()
    # x=dataset.x.clone().detach()
    # y=dataset.y.clone().detach()
    loss = get_loss('classification')
    beta_type = 'other'
    problem_type = 'classification'
    batch_size = 100000
    n_samples = 20
    is_gpu = False
    x = (genome.copy(),
         # x, y,
         loss, beta_type, problem_type,
         batch_size, n_samples, is_gpu)

    tasks.append(x)

# TODO: fix logging when using multiprocessing. Easy fix is to disable
fitnesses = list(pool.imap(evaluate_genome_parallel, tasks, chunksize=max([len(genomes)//N_PROCESSES, 1])))

pool.close()
for i, genome in enumerate(genomes):
    print(fitnesses[i])
    genome.fitness = fitnesses[i]

print('finished')
