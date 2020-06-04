import math

import torch
from torch.nn import Sigmoid, Softmax
from neat.evaluation.utils import _prepare_batch_data
from neat.fitness.kl_divergence import compute_kl_qw_pw
from neat.genome import Genome
from neat.loss.vi_loss import get_beta
from neat.representation_mapping.genome_to_network.complex_stochastic_network import ComplexStochasticNetwork
from neat.representation_mapping.genome_to_network.complex_stochastic_network_jupyneat import \
    ComplexStochasticNetworkJupyneat
from neat.utils import timeit


@timeit
def evaluate_genome(genome: Genome, dataset, loss, beta_type, problem_type, is_testing,
                    batch_size=10000, n_samples=10, is_gpu=False, return_all=False, is_pass=True):
    '''
    Calculates: KL-Div(q(w)||p(w|D))
    Uses the VariationalInferenceLoss class (not the alternative)
    '''
    # kl_posterior = 0

    # kl_qw_pw = compute_kl_qw_pw(genome=genome)

    # setup network
    network = ComplexStochasticNetwork(genome=genome)
    if is_gpu:
        network.cuda()
    m = math.ceil(len(dataset) / batch_size)
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
    chunks_x = []
    chunks_y_pred = []
    chunks_y_true = []
    with torch.no_grad():
        # forward pass
        output, kl_qw_pw = network(x_batch)
        output, _, y_batch = _process_output_data(output, y_true=y_batch, n_samples=n_samples,
                                                  n_output=genome.n_output, problem_type=problem_type, is_pass=is_pass)
        beta = get_beta(beta_type=beta_type, m=m, batch_idx=0, epoch=1, n_epochs=1)
        kl_posterior = loss(y_pred=output, y_true=y_batch, kl_qw_pw=kl_qw_pw, beta=beta)
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
def calculate_prediction_distribution(network, dataset, problem_type, is_testing, n_samples=1000,
                                      use_sigmoid=False):
    '''
    Calculate Predictive Distribution for a network and dataset
    '''
    # setup network
    network.eval()

    # calculate Data log-likelihood (p(y*|x*,D))
    if is_testing:
        x_batch, y_batch = dataset.x_test, dataset.y_test
    else:
        x_batch, y_batch = dataset.x_train, dataset.y_train

    x_batch, y_batch = _prepare_batch_data(x_batch=x_batch,
                                           y_batch=y_batch,
                                           problem_type=problem_type,
                                           is_gpu=False,
                                           n_input=network.n_input,
                                           n_output=network.n_output,
                                           n_samples=n_samples)
    chunks_x = []
    chunks_output_distribution = []
    chunks_y_true = []
    with torch.no_grad():
        # forward pass
        output, _ = network(x_batch)

        _, output_distribution, y_batch = _process_output_data(output, y_true=y_batch, n_samples=n_samples,
                                                               n_output=network.n_output, problem_type=problem_type,
                                                               is_pass=True)

        chunks_x.append(x_batch)
        chunks_output_distribution.append(output_distribution)
        chunks_y_true.append(y_batch)

    x = torch.cat(chunks_x, dim=0)
    output_distribution = torch.cat(chunks_output_distribution, dim=0)
    y_true = torch.cat(chunks_y_true, dim=0)
    return x, y_true, output_distribution


def _process_output_data(output, y_true, n_samples, n_output, problem_type, is_pass=False):

    if not is_pass:
        return output, output, y_true
    n_examples = y_true.shape[0] // n_samples
    y_true = y_true[:n_examples]
    if problem_type == 'classification':
        multinomial, multinomial_dist = calculate_multinomial_3(output, n_samples, n_output)
        return multinomial, multinomial_dist, y_true
    elif problem_type == 'regression':
        output, output_dist = calculate_regression_distribution(output, n_samples, n_output)
        return output, output_dist, y_true
        # raise ValueError('Problem Type not supported yet')
    raise ValueError('Problem Type is wrong')


def calculate_regression_distribution(output, n_samples, n_output):
    logits = _reshape_output(output, n_samples, n_output)
    return logits.mean(1), logits


def convert_to_multinomial(y_pred, n_samples, n_output):
    logits = _reshape_output(y_pred, n_samples, n_output)
    class_prediction_per_sample = torch.argmax(logits, dim=2)
    n_examples = class_prediction_per_sample.shape[0]
    multinomial = torch.zeros(n_examples, n_output)

    for output in range(n_output):
        output_count = (class_prediction_per_sample == output).sum(1)
        multinomial[:, output] = output_count / n_samples
    return multinomial


def convert_to_multinomial_2(y_pred, n_samples, n_output):
    #     logits = y_pred.reshape(n_samples, -1,  n_output).permute(1, 0, 2)
    logits = _reshape_output(y_pred, n_samples, n_output)
    sum_ = logits.sum(2)
    for output in range(n_output):
        logits[:, :, output] = logits[:, :, output] / sum_

    #     class_prediction_per_sample = torch.argmax(, dim=2)
    n_examples = logits.shape[0]
    multinomial = torch.zeros(n_examples, n_output)

    for output in range(n_output):
        output_count = (logits == output).sum(1)
        multinomial[:, output] = output_count / n_samples
    return multinomial


def calculate_multinomial_3(y_pred, n_samples, n_output):
    logits = _reshape_output(y_pred, n_samples, n_output)
    m = Softmax()
    m = Sigmoid()
    multinomial_dist = m(logits)
    sum_ = multinomial_dist.sum(2)
    for output in range(n_output):
        multinomial_dist[:, :, output] = multinomial_dist[:, :, output] / sum_
    return multinomial_dist.mean(1), multinomial_dist


def calculate_multinomial(y_pred, n_samples, n_output):
    logits = _reshape_output(y_pred, n_samples, n_output)
    m = Softmax(dim=2)
    output = m(logits)
    return output.mean(1), output


def _reshape_output(output, n_samples, n_output):
    return output.reshape(n_samples, -1, n_output).permute(1, 0, 2)

@timeit
def evaluate_genome_jupyneat(genome: dict, dataset, loss, beta_type, problem_type, n_input, n_output, activation_type,
                             batch_size=10000, n_samples=10, is_gpu=False, return_all=False):
    '''
    Calculates: KL-Div(q(w)||p(w|D))
    Uses the VariationalInferenceLoss class (not the alternative)
    '''
    kl_posterior = 0

    # kl_qw_pw = compute_kl_qw_pw(genome=genome)
    kl_qw_pw = 0.0

    # setup network
    network = ComplexStochasticNetworkJupyneat(genome=genome, n_input=n_input, n_output=n_output,
                                               activation_type=activation_type)
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
                                           n_input=n_input,
                                           n_output=n_output,
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
