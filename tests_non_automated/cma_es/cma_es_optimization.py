from cma import CMAEvolutionStrategy
import numpy as np

from neat.evaluation import EvaluationStochasticEngine
from tests.config_files.config_files import create_configuration
from tests.utils.generate_genome import generate_genome_with_hidden_units


config_file = '/classification-miso.json'
# config_file = '/siso.json'

config = create_configuration(filename=config_file)
N_SAMPLES = 100

n_neurons_per_layer = 10

total_neurons = n_neurons_per_layer + config.n_input + config.n_output
total_biases = total_neurons
total_bias_params = total_biases * 2

total_weights = config.n_input * n_neurons_per_layer + n_neurons_per_layer * config.n_output
iteration = 0


def regression_problem(x):

    bias = x[:total_bias_params]
    weight = x[total_bias_params:]

    genome = generate_genome_with_hidden_units(n_input=config.n_input,
                                               n_output=config.n_output,
                                               n_hidden=n_neurons_per_layer)
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

    evaluation_engine = EvaluationStochasticEngine(batch_size=50000)

    loss = \
        evaluation_engine.evaluate_genome(genome=genome, n_samples=N_SAMPLES, return_all=False)
    print(loss)
    return loss


# def regression_problem(x):
#     print(0.5)
#     return 0.5

def main():

    bias0 = [0.0, -3.0] * total_biases
    weight0 = [0.0, -3.0] * total_weights
    x0 = bias0
    x0.extend(weight0)

    # assert(len(x0) == 108)
    sigma0 = 0.1

    # evolution strategy
    es = CMAEvolutionStrategy(x0, sigma0)
    optSol = es.optimize(regression_problem)
    print(optSol)

    sol = [-3.18979598e-02,  1.21945777e+00,  3.80408518e-02,
            7.99747577e-01,  9.51714572e-02,  8.10528673e-01, -5.97132372e-02,
            1.04204663e+00,  1.74305244e-02,  9.93532532e-01, -8.67464165e-02,
            9.51615254e-01, -3.77311791e-02,  9.97444596e-01, -5.84591048e-03,
            1.08793924e+00, -5.59697903e-02,  9.51347690e-01,  3.33323542e-02,
            1.00693954e+00]


def example_cma():
    def yangs4(x):
        # f = (a-b).c
        a = np.sum(np.square(np.sin(x)))
        b = np.exp(-np.sum(np.square(x)))
        c = np.exp(-np.sum(np.square(np.sin(np.sqrt(np.abs(x))))))
        return (a - b) * c

    n = 5
    x0 = n * [0.1]
    sigma0 = 0.1
    # evolution strategy
    es = CMAEvolutionStrategy(x0, sigma0)
    optSol = es.optimize(yangs4)


if __name__ == '__main__':
    main()
