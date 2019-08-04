import torch

from neat.fitness.kl_divergence import compute_kl_qw_pw, compute_kl_qw_pw_by_product
from tests_non_automated.deep_learning.feed_forward import FeedForward
from neat.evaluation import EvaluationEngine
from neat.representation.stochastic_network import StochasticNetwork
from tests.config_files.config_files import create_configuration
from tests.utils.generate_genome import generate_genome_with_hidden_units
import matplotlib.pyplot as plt

from tests_non_automated.evaluate_solution import prepare_genome

config = create_configuration(filename='/siso.json')
N_SAMPLES = 50
std = 0.2
n_neurons_per_layer = 10


def main():

    # standard network
    network = FeedForward(n_input=config.n_input, n_output=config.n_output,
                          n_neurons_per_layer=n_neurons_per_layer,
                          n_hidden_layers=1)
    parameters = torch.load('./../deep_learning/models/network.pt')
    network.load_state_dict(parameters)

    genome = generate_genome_with_hidden_units(n_input=config.n_input,
                                               n_output=config.n_output,
                                               n_hidden=n_neurons_per_layer)

    # genome = prepare_genome(parameters)
    print(genome)

    print('KL (q(w)||p(w)')
    kl_qw_pw = compute_kl_qw_pw(genome=genome)
    print(kl_qw_pw)


    evaluation_engine = EvaluationEngine(testing=False, batch_size=1)

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


if __name__ == '__main__':
    main()
