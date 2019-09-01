import torch
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score
from neat.evaluation import EvaluationStochasticGoodEngine
from neat.representation_mapping.network_to_genome.standard_feed_forward_to_genome import \
    get_genome_from_standard_network
from tests.config_files.config_files import create_configuration
from deep_learning.standard.feed_forward import FeedForward

config = create_configuration(filename='/classification-miso.json')
N_SAMPLES = 50
std = 0.2
n_neurons_per_layer = 10


def main():
    model_filename = 'network-classification.pt'
    network = FeedForward(n_input=config.n_input, n_output=config.n_output,
                          n_neurons_per_layer=n_neurons_per_layer,
                          n_hidden_layers=1)
    parameters = torch.load(f'./../../deep_learning/models/{model_filename}')
    network.load_state_dict(parameters)

    std = -2.1
    genome = get_genome_from_standard_network(network, std=std)

    # genome = generate_genome_with_hidden_units(2, 2, n_hidden=1)

    evaluation_engine = EvaluationStochasticGoodEngine(testing=False)

    x, y_true, y_pred, kl_posterior = \
        evaluation_engine.evaluate_genome(genome, n_samples=100, is_gpu=False, return_all=True)

    print()
    print(f'KL Posterior: {kl_posterior}')

    x = x.numpy()
    x = evaluation_engine.dataset.input_scaler.inverse_transform(x)
    y_true = y_true.numpy()

    # plot results
    y_pred = np.argmax(y_pred.numpy(), 1)
    df = pd.DataFrame(x, columns=['x1', 'x2'])
    df['y'] = y_pred

    x1_limit, x2_limit = evaluation_engine.dataset.get_separation_line()

    plt.figure()
    ax = sns.scatterplot(x='x1', y='x2', hue='y', data=df)
    ax.plot(x1_limit, x2_limit, 'g-', linewidth=2.5)
    plt.show()


    print('Confusion Matrix:')
    print(confusion_matrix(y_true, y_pred))

    print(f'Accuracy: {accuracy_score(y_true, y_pred) * 100} %')

if __name__ == '__main__':
    main()