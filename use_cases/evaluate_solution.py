import torch
import numpy as np
from neat.evaluation import EvaluationEngine
from neat.representation.stochastic_network import StochasticNetwork
from tests.config_files.config_files import create_configuration
from tests.utils.generate_genome import generate_genome_with_hidden_units
import matplotlib.pyplot as plt

config = create_configuration(filename='/siso.json')
N_SAMPLES = 100


def regression_problem(x):
    bias = x[:8]
    weight = x[8:]

    genome = generate_genome_with_hidden_units(n_input=config.n_input,
                                               n_output=config.n_output)
    # nodes = genome.node_genes
    # connections = genome.connection_genes
    #
    # i = 0
    # for key, node in nodes.items():
    #     node.bias_mean = bias[i]
    #     nodes[0].bias_std = bias[i+1]
    #     i += 2
    #
    # i = 0
    # for key, connection in connections.items():
    #     connection.weight_mean = weight[i]
    #     connection.weight_std = weight[i + 1]
    #     i += 2

    evaluation_engine = EvaluationEngine(testing=True)
    data_loader = evaluation_engine.data_loader
    loss = evaluation_engine.loss
    # setup network
    network = StochasticNetwork(genome=genome)
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
        y_true = y_batch.numpy()
        y_pred = output.numpy().reshape(-1)
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

    loss = regression_problem(x)
    print(loss)


if __name__ == '__main__':
    main()