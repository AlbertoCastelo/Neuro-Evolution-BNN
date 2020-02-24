import torch
from experiments.reporting.report_repository import ReportRepository
from neat.evaluation.evaluate_parallel import _evaluate_genome_parallel, process_initialization
from neat.evaluation.evaluation_engine import evaluate_genome, get_dataset
from neat.genome import Genome
import os
from neat.loss.vi_loss import get_loss
from neat.neat_logger import get_neat_logger


LOGS_PATH = f'{os.getcwd()}/'
logger = get_neat_logger(path=LOGS_PATH)


def main():
    ALGORITHM_VERSION = 'bayes-neat'
    DATASET = 'titanic'
    CORRELATION_ID = 'tests'
    execution_id = '438fc562-0ff0-457b-9c20-7c3bd7003b7c'   # fix architecture
    execution_id = '5e40afd0-2519-4d2e-813b-7023a730b1af'   # learning architecture 80 % of data
    execution_id = 'bfee8549-4d76-426f-8215-f2de25ce794f'   # learning architecture 40 % of data

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
