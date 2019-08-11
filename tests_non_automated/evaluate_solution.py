import torch
import numpy as np

from neat.representation.stochastic_network import StochasticNetworkOld, StochasticNetwork
from tests_non_automated.deep_learning.feed_forward import FeedForward
from neat.evaluation import EvaluationEngine, EvaluationStochasticGoodEngine
from tests.config_files.config_files import create_configuration
from tests.utils.generate_genome import generate_genome_with_hidden_units
import matplotlib.pyplot as plt

config = create_configuration(filename='/siso.json')
N_SAMPLES = 50
# std = -2.57
std = 0.0001
n_neurons_per_layer = 10


def regression_problem_learn_from_nn():

    # standard network
    network = FeedForward(n_input=config.n_input, n_output=config.n_output,
                          n_neurons_per_layer=n_neurons_per_layer,
                          n_hidden_layers=1)
    parameters = torch.load('./deep_learning/models/network.pt')
    network.load_state_dict(parameters)

    genome = prepare_genome(parameters)
    print(genome)
    evaluation_engine = EvaluationStochasticGoodEngine(testing=False, batch_size=100000)

    # setup network
    network = StochasticNetwork(genome=genome)
    network.eval()

    x, y_true, y_pred, kl_posterior, kl_qw_pw = \
        evaluation_engine.evaluate_genome(genome, n_samples=10, is_gpu=False, return_all=True)

    plt.figure(figsize=(20, 20))
    plt.plot(x.numpy().reshape(-1), y_true.numpy().reshape(-1), 'b*')
    plt.plot(x.numpy().reshape(-1), y_pred.numpy().reshape(-1), 'r*')
    plt.show()
    print(f'KL Div - Posterior: {kl_posterior}')
    print(f'KL Div - Prior: {kl_qw_pw}')
    print(f'MSE: {kl_posterior - kl_qw_pw}')


def prepare_genome(parameters):
    genome = generate_genome_with_hidden_units(n_input=config.n_input,
                                               n_output=config.n_output,
                                               n_hidden=n_neurons_per_layer)
    nodes = genome.node_genes
    connections = genome.connection_genes

    # output nodes
    output_nodes = [0]
    for output_key in output_nodes:
        nodes[output_key].bias_mean = float(parameters['layer_0.bias'].numpy()[output_key])
        nodes[output_key].bias_std = std

    # hidden nodes
    i = 0
    bias_1 = parameters['layer_1.bias'].numpy()
    for key, node in nodes.items():
        if key in output_nodes:
            continue
        node.bias_mean = float(bias_1[i])
        node.bias_std = std
        i += 1

    # WEIGHTS
    hidden_neurons = list(range(1, 11))
    # layer 1 (input-> hidden)
    weights_1 = parameters['layer_1.weight'].numpy()
    for key, connection in connections.items():
        if key[1] not in hidden_neurons:
            continue
        connection.weight_mean = float(weights_1[key[1] - 1, 0])
        connection.weight_std = std

    weights_0 = parameters['layer_0.weight'].numpy()
    for key, connection in connections.items():
        if key[0] not in hidden_neurons:
            continue
        connection.weight_mean = float(weights_0[0, key[0] - 1])
        connection.weight_std = std
    return genome


def regression_problem(x):
    bias = x[:8]
    weight = x[8:]

    genome = generate_genome_with_hidden_units(n_input=config.n_input,
                                               n_output=config.n_output)
    nodes = genome.node_genes
    connections = genome.connection_genes

    i = 0
    for key, node in nodes.items():
        node.bias_mean = bias[i]
        node.bias_std = bias[i+1]
        i += 2

    i = 0
    for key, connection in connections.items():
        connection.weight_mean = weight[i]
        connection.weight_std = weight[i + 1]
        i += 2

    evaluation_engine = EvaluationEngine(testing=True)
    dataset = evaluation_engine.dataset
    data_loader = evaluation_engine.data_loader
    loss = evaluation_engine.loss
    # setup network
    network = StochasticNetworkOld(genome=genome)
    network.eval()

    # calculate Data log-likelihood (p(y*|x*,D))
    mses = []
    xs = []
    y_preds = []
    y_trues = []
    for i in range(N_SAMPLES):
        for x_batch, y_batch in data_loader:
            x_batch = x_batch.reshape((-1, genome.n_input))
            x_batch = x_batch.float()
            y_batch = y_batch.float()

            with torch.no_grad():
                # forward pass
                output = network(x_batch)

                mse = loss(y_pred=output, y_true=y_batch, kl_qw_pw=0, beta=evaluation_engine.get_beta())
                mses.append(mse)
            x = x_batch.numpy().reshape(-1)
            y_true_unnormalized = dataset.unnormalize_output(y_pred=y_batch)
            y_true = y_true_unnormalized.numpy()
            output_unnormalized = dataset.unnormalize_output(y_pred=output)
            y_pred = output_unnormalized.numpy().reshape(-1)
            xs.append(x)
            y_trues.append(y_true)
            y_preds.append(y_pred)
    x = np.concatenate(xs)
    y_true = np.concatenate(y_trues)
    y_pred = np.concatenate(y_preds)
    plt.figure(figsize=(20, 20))
    plt.plot(x, y_true, 'b*')
    plt.plot(x, y_pred, 'r*')
    plt.show()
    print(mse)


def main():
    x = [-3.18979598e-02, 1.21945777e+00, 3.80408518e-02,
       7.99747577e-01, 9.51714572e-02, 8.10528673e-01, -5.97132372e-02,
       1.04204663e+00, 1.74305244e-02, 9.93532532e-01, -8.67464165e-02,
       9.51615254e-01, -3.77311791e-02, 9.97444596e-01, -5.84591048e-03,
       1.08793924e+00, -5.59697903e-02, 9.51347690e-01, 3.33323542e-02,
       1.00693954e+00]

    x_pymc3 = [0.647056, 0.002242, 0.005813, 0.277180, 2.153285, 0.056461, 0.005075, 0.186190,
               -0.011444, 0.302296, 0.003719, 0.057393, -0.006722, 0.201250,
               -0.001349, 0.008320, -0.665375, 0.002420, 0.003374, 0.011338]

    loss = regression_problem(x_pymc3)
    print(loss)


if __name__ == '__main__':
    regression_problem_learn_from_nn()