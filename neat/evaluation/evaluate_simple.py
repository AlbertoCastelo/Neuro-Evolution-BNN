import math

import torch

from neat.evaluation.utils import _prepare_batch_data
from neat.fitness.kl_divergence import compute_kl_qw_pw
from neat.genome import Genome
from neat.loss.vi_loss import get_beta
from neat.representation_mapping.genome_to_network.complex_stochastic_network import ComplexStochasticNetwork
from neat.utils import timeit


@timeit
def evaluate_genome(genome: Genome, dataset, loss, beta_type, problem_type,
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
    m = math.ceil(len(dataset) / batch_size)
    network.eval()

    chunks_x = []
    chunks_y_pred = []
    chunks_y_true = []

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
        beta = get_beta(beta_type=beta_type, m=m, batch_idx=0, epoch=1, n_epochs=1)
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
def evaluate_genome_with_dataloader(genome: Genome, data_loader, loss, beta_type, problem_type,
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
        print(len(x_batch))
        print(f'x_batch sum: {x_batch.sum()}')
        with torch.no_grad():
            # forward pass
            output, _ = network(x_batch)
            print(f'Output sum: {output.sum()}')
            beta = get_beta(beta_type=beta_type, m=m, batch_idx=batch_idx, epoch=1, n_epochs=1)
            print(f'Beta: {beta}')
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
