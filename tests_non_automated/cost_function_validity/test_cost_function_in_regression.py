import torch
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from neat.evaluation import EvaluationStochasticGoodEngine
from neat.representation_mapping.network_to_genome.standard_feed_forward_to_genome import \
    get_genome_from_standard_network
from tests.config_files.config_files import create_configuration
from deep_learning.feed_forward import FeedForward

config = create_configuration(filename='/siso.json')
N_SAMPLES = 50
std = 0.2
n_neurons_per_layer = 10


def main():
    model_filename = 'network-regression_1.pt'
    network = FeedForward(n_input=config.n_input, n_output=config.n_output,
                          n_neurons_per_layer=n_neurons_per_layer,
                          n_hidden_layers=1)
    parameters = torch.load(f'./../deep_learning/models/{model_filename}')
    network.load_state_dict(parameters)

    std = -3.1
    genome = get_genome_from_standard_network(network, std=std)

    evaluation_engine = EvaluationStochasticGoodEngine(testing=False)


    x, y_true, y_pred, kl_posterior = \
        evaluation_engine.evaluate_genome(genome, n_samples=1, is_gpu=False, return_all=True)

    print()
    print(f'KL Posterior: {kl_posterior}')


    x = x.numpy()
    x = evaluation_engine.dataset.input_scaler.inverse_transform(x)
    y_pred = evaluation_engine.dataset.output_scaler.inverse_transform(y_pred.numpy())
    y_true = evaluation_engine.dataset.output_scaler.inverse_transform(y_true.numpy())

    # plot results
    plt.figure(figsize=(20, 20))
    plt.plot(x, y_true, 'r*')
    plt.plot(x, y_pred, 'b*')
    plt.show()

    print(f'MSE: {mean_squared_error(y_true, y_pred) * 100} %')

if __name__ == '__main__':
    main()