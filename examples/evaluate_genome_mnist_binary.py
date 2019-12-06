import torch
from torch.utils.data import DataLoader

from experiments.reporting.report_repository import ReportRepository
from neat.evaluation.evaluate_parallel import _evaluate_genome_parallel, process_initialization
from neat.evaluation.evaluation_engine import evaluate_genome, get_dataset
from neat.genome import Genome
import os
from neat.loss.vi_loss import get_loss
from neat.neat_logger import get_neat_logger
from tests.config_files.config_files import create_configuration

# dataset = 'mnist_binary'
# config = create_configuration(filename=f'/{dataset}.json')

LOGS_PATH = f'{os.getcwd()}/'
logger = get_neat_logger(path=LOGS_PATH)


def main():
    ALGORITHM_VERSION = 'bayes-neat'
    DATASET = 'mnist_binary'
    CORRELATION_ID = 'tests'
    execution_id = 'f6d2d5e3-26a3-4069-9071-b74009323761' # 2 hours run
    # execution_id = '59cbe09c-4ee7-4e7e-9b17-26c866113cfe' # test-run
    # execution_id = 'c5551a6c-177b-4c2c-8ecd-a75e79ae0ec2'
    report_repository = ReportRepository.create(project='neuro-evolution', logs_path=LOGS_PATH)
    report = report_repository.get_report(algorithm_version=ALGORITHM_VERSION,
                                          dataset=DATASET,
                                          correlation_id=CORRELATION_ID,
                                          execution_id=execution_id)
    genome_dict = report.data['best_individual']
    best_individual_fitness = report.data['best_individual_fitness']
    print(f'Fitness of best individual: {best_individual_fitness}')
    # genome_filename = f'./executions/11-02-2019, 19:12:46fcfd8573-5275-48ef-a92b-2ec41050c0de.json'
    # genome_dict = read_json_file_to_dict(filename=genome_filename)
    genome = Genome.from_dict(genome_dict)
    config = genome.genome_config

    ##### EVALUATE ######
    loss = get_loss(problem_type=config.problem_type)
    evaluate_with_parallel(genome, loss, config)

    is_cuda = False

    dataset = get_dataset(config.dataset_name, testing=True)
    dataset.generate_data()
    data_loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True, num_workers=4)

    x, y_true, y_pred, loss_value = evaluate_genome(genome=genome,
                                                    data_loader=data_loader,
                                                    loss=loss,
                                                    problem_type=config.problem_type,
                                                    beta_type=config.beta_type,
                                                    batch_size=config.batch_size,
                                                    n_samples=config.n_samples,
                                                    is_gpu=is_cuda,
                                                    return_all=True)
    y_pred = torch.argmax(y_pred, dim=1)
    # predict
    print('Evaluating results')

    from sklearn.metrics import confusion_matrix, accuracy_score
    print(f'Loss: {loss_value}')
    print('Confusion Matrix:')
    print(confusion_matrix(y_true, y_pred))

    print(f'Accuracy: {accuracy_score(y_true, y_pred) * 100} %')

    # plot_genome_network(genome, view=True)


def evaluate_with_parallel(genome, loss, config):

    process_initialization(dataset_name=config.dataset_name, testing=True)
    loss_value = _evaluate_genome_parallel(genome=genome, loss=loss, beta_type=config.beta_type,
                                           problem_type=config.problem_type,
                                           batch_size=100000, n_samples=20)
    print(f'Parallel loss: {loss_value}')


if __name__ == '__main__':
    main()
