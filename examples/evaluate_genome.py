from torch.utils.data import DataLoader

from neat.configuration import read_json_file_to_dict
from neat.evaluation import evaluate_genome, get_dataset
from neat.genome import Genome
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from neat.loss.vi_loss import get_loss
from neat.plotting.plot_network import plot_network, plot_genome_network
from tests.config_files.config_files import create_configuration

config_file = '/classification-miso.json'
config = create_configuration(filename=config_file)


def main():

    genome_filename = f'./executions/11-02-2019, 19:12:46fcfd8573-5275-48ef-a92b-2ec41050c0de.json'
    genome_dict = read_json_file_to_dict(filename=genome_filename)
    genome = Genome.from_dict(genome_dict)

    is_cuda = False

    dataset = get_dataset(config.dataset_name, testing=False)
    dataset.generate_data()
    data_loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True, num_workers=4)

    loss = get_loss(problem_type=config.problem_type)

    x, y_true, y_pred, loss_value = evaluate_genome(genome=genome, data_loader=data_loader, loss=loss,
                                                    beta_type=config.beta_type,
                                                    batch_size=10000, n_samples=100, is_gpu=is_cuda, return_all=True)

    # predict
    print('Evaluating results')

    if is_cuda:
        x = x.cpu()
        y_true = y_true.cpu()
        y_pred = y_pred.cpu()
    x = dataset.input_scaler.inverse_transform(x)
    # y_true =y_true.numpy()

    # plot results

    y_pred = np.argmax(y_pred.numpy(), 1)
    df = pd.DataFrame(x, columns=['x1', 'x2'])
    df['y'] = y_pred

    x1_limit, x2_limit = dataset.get_separation_line()

    plt.figure()
    ax = sns.scatterplot(x='x1', y='x2', hue='y', data=df)
    ax.plot(x1_limit, x2_limit, 'g-', linewidth=2.5)
    plt.show()

    from sklearn.metrics import confusion_matrix, accuracy_score
    print(f'Loss: {loss_value}')
    print('Confusion Matrix:')
    print(confusion_matrix(y_true, y_pred))

    print(f'Accuracy: {accuracy_score(y_true, y_pred) * 100} %')

    plot_genome_network(genome, view=True)
    # plot_network(nodes=list(genome.node_genes.keys()),
    #              edges=list(genome.connection_genes.keys()),
    #              input_nodes=genome.get_input_nodes_keys(),
    #              output_nodes=genome.get_output_nodes_keys(),
    #              view=True)

if __name__ == '__main__':
    main()
