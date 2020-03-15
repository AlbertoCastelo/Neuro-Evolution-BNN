import torch
from experiments.reporting.report_repository import ReportRepository
from neat.evaluation.evaluate_parallel import _evaluate_genome_parallel, process_initialization
from neat.evaluation.evaluation_engine import evaluate_genome, get_dataset
from neat.genome import Genome
import os
from neat.loss.vi_loss import get_loss
from neat.neat_logger import get_neat_logger
from config_files import create_configuration

LOGS_PATH = f'{os.getcwd()}/'
logger = get_neat_logger(path=LOGS_PATH)

config_file = 'mnist_binary'
config = create_configuration(filename=f'/{config_file}.json')


def main():
    ALGORITHM_VERSION = 'bayes-neat'
    DATASET = 'mnist_binary'
    CORRELATION_ID = 'test'
    # execution_id = 'f6d2d5e3-26a3-4069-9071-b74009323761' # 2 hours run
    execution_id = 'bf516f54-c29b-4f88-949c-102ab67930b3' # 10 hours run (learning architecture)
    # execution_id = '59cbe09c-4ee7-4e7e-9b17-26c866113cfe' # test-run
    # execution_id = 'c5551a6c-177b-4c2c-8ecd-a75e79ae0ec2'
    execution_id = '1f30c172-9056-4012-9651-0765527bd550'  # fitness -0.2
    execution_id = 'a91761a0-6201-4a1d-9293-5e713f305fbf' # fitness -0.86
    execution_id = '991b275d-6282-4f7d-8e97-3908baf94726'
    report_repository = ReportRepository.create(project='neuro-evolution', logs_path=LOGS_PATH)
    report = report_repository.get_report(algorithm_version=ALGORITHM_VERSION,
                                          dataset=DATASET,
                                          correlation_id=CORRELATION_ID,
                                          execution_id=execution_id)
    genome_dict = report.data['best_individual']
    best_individual_fitness = report.data['best_individual_fitness']
    print(f'Fitness of best individual: {best_individual_fitness}')

    genome = Genome.create_from_julia_dict(genome_dict)
    # config = get_configuration()
    print(f'Execution id: {execution_id}')

    loss = get_loss(problem_type=config.problem_type)

    ##### EVALUATE ######
    print('Evaluating results')
    evaluate_with_parallel(genome, loss, config)

    dataset = get_dataset(config.dataset, testing=True)
    dataset.generate_data()
    # TODO: remove data-loader. If we want to sample the dataset in each generation, the we can create a
    #  middlelayer between evaluation and dataset
    x, y_true, y_pred, loss_value = evaluate_genome(genome=genome,
                                                    dataset=dataset,
                                                    loss=loss,
                                                    problem_type=config.problem_type,
                                                    beta_type=config.beta_type,
                                                    batch_size=config.batch_size,
                                                    n_samples=config.n_samples,
                                                    is_gpu=config.is_gpu,
                                                    return_all=True)
    y_pred = torch.argmax(y_pred, dim=1)

    from sklearn.metrics import confusion_matrix, accuracy_score
    print(f'Loss: {loss_value}')
    print('Confusion Matrix:')
    print(confusion_matrix(y_true, y_pred))

    print(f'Accuracy: {accuracy_score(y_true, y_pred) * 100} %')

    # plot_genome_network(genome, view=True)


def evaluate_with_parallel(genome, loss, config):

    process_initialization(dataset_name=config.dataset, testing=True)
    loss_value = _evaluate_genome_parallel(genome=genome, loss=loss, beta_type=config.beta_type,
                                           problem_type=config.problem_type,
                                           batch_size=config.batch_size, n_samples=config.n_samples)
    print(f'Parallel loss: {loss_value}')


if __name__ == '__main__':
    main()
