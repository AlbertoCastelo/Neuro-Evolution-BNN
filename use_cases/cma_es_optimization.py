import cma
from cma import CMAEvolutionStrategy
import numpy as np

from neat.evaluation import EvaluationEngine
from tests.config_files.config_files import create_configuration
from tests.utils.generate_genome import generate_genome_with_hidden_units

config = create_configuration(filename='/siso.json')
N_SAMPLES = 10


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
        nodes[0].bias_std = bias[i+1]
        i += 2

    i = 0
    for key, connection in connections.items():
        connection.weight_mean = weight[i]
        connection.weight_std = weight[i + 1]
        i += 2

    evaluation_engine = EvaluationEngine()

    loss = evaluation_engine.evaluate_genome(genome=genome, n_samples=N_SAMPLES)
    return loss


def main():

    bias0 = [0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0]
    weight0 = [0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0]
    x0 = bias0
    x0.extend(weight0)
    assert(len(x0) == 20)
    sigma0 = 0.1
    # evolution strategy
    es = CMAEvolutionStrategy(x0, sigma0)
    optSol = es.optimize(regression_problem)
    print(optSol)

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