from torch.utils.data import DataLoader

from experiments.object_repository.object_repository import ObjectRepository
from experiments.reporting.report_repository import ReportRepository
from neat.configuration import read_json_file_to_dict
from neat.evaluation import evaluate_genome, get_dataset
from neat.genome import Genome
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
from neat.loss.vi_loss import get_loss
from neat.plotting.plot_network import plot_network, plot_genome_network
from tests.config_files.config_files import create_configuration

dataset = 'mnist_binary'
config = create_configuration(filename=f'/{dataset}.json')
LOGS_PATH = f'{os.getcwd()}/'


def main():
    ALGORITHM_VERSION = 'bayes-neat'
    DATASET = 'mnist'
    CORRELATION_ID = 'tests'
    execution_id = '37aee733-ba1c-4180-afe6-e66b24546f98'
    report_repository = ReportRepository.create(project='neuro-evolution', logs_path=LOGS_PATH)
    report = report_repository.get_report(algorithm_version=ALGORITHM_VERSION,
                                          dataset=DATASET,
                                          correlation_id=CORRELATION_ID,
                                          execution_id=execution_id)
    genome_dict = None

    # genome_filename = f'./executions/11-02-2019, 19:12:46fcfd8573-5275-48ef-a92b-2ec41050c0de.json'
    # genome_dict = read_json_file_to_dict(filename=genome_filename)
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

    from sklearn.metrics import confusion_matrix, accuracy_score
    print(f'Loss: {loss_value}')
    print('Confusion Matrix:')
    print(confusion_matrix(y_true, y_pred))

    print(f'Accuracy: {accuracy_score(y_true, y_pred) * 100} %')

    plot_genome_network(genome, view=True)

if __name__ == '__main__':
    main()
