import random
import matplotlib.pyplot as plt
import torch
from experiments.reporting.report_repository import ReportRepository
from neat.configuration import set_configuration
from neat.evaluation.evaluate_parallel import _evaluate_genome_parallel, process_initialization
from neat.evaluation.evaluation_engine import evaluate_genome, get_dataset
from neat.genome import Genome
import os
from neat.loss.vi_loss import get_loss
from neat.neat_logger import get_neat_logger
from neat.plotting.plot_network import plot_genome_network


LOGS_PATH = f'{os.getcwd()}/'
logger = get_neat_logger(path=LOGS_PATH)


def main():
    ALGORITHM_VERSION = 'bayes-neat'
    DATASET = 'mnist_binary'
    CORRELATION_ID = 'tests'
    # execution_id = 'f6d2d5e3-26a3-4069-9071-b74009323761' # 2 hours run
    # execution_id = 'bf516f54-c29b-4f88-949c-102ab67930b3' # 10 hours run (learning architecture)
    # execution_id = '59cbe09c-4ee7-4e7e-9b17-26c866113cfe' # test-run
    # execution_id = 'c5551a6c-177b-4c2c-8ecd-a75e79ae0ec2'
    execution_id = '1f30c172-9056-4012-9651-0765527bd550'  # fitness -0.2
    # execution_id = 'a91761a0-6201-4a1d-9293-5e713f305fbf'    # fitness -0.86
    # execution_id = 'eaa675cc-bb02-4a03-ad8f-fbe40b04762a'
    # execution_id = '4a0aba36-34b3-4d28-b536-ef01734505cc'
    # execution_id = 'edf899f5-d231-4567-a40e-59d9967013d4'
    # execution_id = '855fd0c8-5fd8-4b75-a39c-c678f690821f'
    # execution_id = '3c086424-bc94-438d-9ea2-d37a3021d8de'
    # execution_id = 'c19dc3d5-735a-44a7-809d-71da51696237'

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
    config.dataset = 'mnist_binary'
    set_configuration(config)
    print(f'Execution id: {execution_id}')

    loss = get_loss(problem_type=config.problem_type)

    ##### EVALUATE ######
    print('Evaluating results')
    evaluate_with_parallel(genome, loss, config)

    dataset = get_dataset(config.dataset, testing=True)

    selection = random.choice(list(range(len(dataset))))
    print(selection)
    x, y = dataset.__getitem__(selection)
    x = x.squeeze().numpy()
    plt.imshow(x)
    plt.show()
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
