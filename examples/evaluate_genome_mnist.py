import torch
from experiments.reporting.report_repository import ReportRepository
from neat.evaluation.evaluate_parallel import _evaluate_genome_parallel, process_initialization
from neat.evaluation.evaluation_engine import evaluate_genome, get_dataset
from neat.genome import Genome
import os
from neat.loss.vi_loss import get_loss
from neat.neat_logger import get_neat_logger
from neat.analysis.plotting.plot_network import plot_genome_network


LOGS_PATH = f'{os.getcwd()}/'
logger = get_neat_logger(path=LOGS_PATH)


def main():
    ALGORITHM_VERSION = 'bayes-neat'

    CORRELATION_ID = 'test'
    # DATASET = 'mnist'
    # # execution_id = 'f6d2d5e3-26a3-4069-9071-b74009323761' # 2 hours run
    # execution_id = 'bf516f54-c29b-4f88-949c-102ab67930b3' # 10 hours run (learning architecture)
    # # execution_id = '59cbe09c-4ee7-4e7e-9b17-26c866113cfe' # test-run
    # # execution_id = 'c5551a6c-177b-4c2c-8ecd-a75e79ae0ec2'
    # execution_id = 'a70103cb-e5f5-4c35-b3d6-5ab4e21e96ac'  # structure only has 1 output node
    # execution_id = '440c2b43-f514-4d9b-873b-47436ee96137'  # initial network creates connection for all output nodes
    # execution_id = 'c65d7ee2-135f-4527-adb4-bfc84e66f933'

    DATASET = 'mnist_downsampled'
    execution_id = '58dc7890-969b-4643-b002-8757b4054260'
    execution_id = '15a2cf07-01db-42e1-83f4-a4c5ca2a13c9'
    execution_id = '177d1879-9ffb-4c95-9895-503256edf7aa'
    execution_id = 'f9ba67cf-be7b-44f4-b271-7264c901b642'
    execution_id = '19ca821b-9d34-415d-9b5e-fff24adee132'
    execution_id = 'cb3a361c-922a-4fd1-9368-1fc341cc9d1f'

    report_repository = ReportRepository.create(project='neuro-evolution', logs_path=LOGS_PATH)
    report = report_repository.get_report(algorithm_version=ALGORITHM_VERSION,
                                          dataset=DATASET,
                                          correlation_id=CORRELATION_ID,
                                          execution_id=execution_id)
    genome_dict = report.data['best_individual']
    best_individual_fitness = report.data['best_individual_fitness']
    print(f'Fitness of best individual: {best_individual_fitness}')

    genome = Genome.from_dict(genome_dict)
    config = genome.genome_config
    print(f'Execution id: {execution_id}')

    loss = get_loss(problem_type=config.problem_type)

    ##### EVALUATE ######
    print('Evaluating results')
    evaluate_with_parallel(genome, loss, config, is_testing=False)

    dataset = get_dataset(config.dataset, testing=True)
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

    plot_genome_network(genome, view=True)


def evaluate_with_parallel(genome, loss, config, is_testing):

    process_initialization(dataset_name=config.dataset, testing=True)
    loss_value = _evaluate_genome_parallel(genome=genome, loss=loss, beta_type=config.beta_type,
                                           problem_type=config.problem_type,
                                           batch_size=config.batch_size, n_samples=config.n_samples,
                                           is_testing=is_testing)
    print(f'Parallel loss: {loss_value}')


if __name__ == '__main__':
    main()
