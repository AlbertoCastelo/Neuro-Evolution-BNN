import jsons
import torch

from experiments.reporting.report_repository import ReportRepository
from neat.configuration import BaseConfiguration
from neat.evaluation.evaluate_simple import evaluate_genome_jupyneat
from neat.evaluation.evaluation_engine import get_dataset
import os
from neat.loss.vi_loss import get_loss
from neat.neat_logger import get_neat_logger

# config_file = '/classification-miso.json'
# config = create_configuration(filename=config_file)


LOGS_PATH = f'{os.getcwd()}/'
logger = get_neat_logger(path=LOGS_PATH)


def main():
    ALGORITHM_VERSION = 'bayes-neat'
    DATASET = 'classification_example_1'
    CORRELATION_ID = 'test'
    execution_id = '180186eb-46c8-4bbd-9f8a-26a36cbe57e4'

    report_repository = ReportRepository.create(project='neuro-evolution', logs_path=LOGS_PATH)
    report = report_repository.get_report(algorithm_version=ALGORITHM_VERSION,
                                          dataset=DATASET,
                                          correlation_id=CORRELATION_ID,
                                          execution_id=execution_id)
    genome = report.data['best_individual']
    best_individual_fitness = report.data['best_individual_fitness']
    print(f'Fitness of best individual: {best_individual_fitness}')

    config_dict = report.config
    config = jsons.load(config_dict, BaseConfiguration)

    loss = get_loss(problem_type=config.problem_type)
    dataset = get_dataset(config.dataset, testing=True)
    dataset.generate_data()

    x, y_true, y_pred, loss_value = evaluate_genome_jupyneat(genome=genome,
                                                             problem_type=config.problem_type,
                                                             n_input=config.n_input,
                                                             n_output=config.n_output,
                                                             activation_type=config.node_activation,
                                                             dataset=dataset,
                                                             loss=loss,
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


if __name__ == '__main__':
    main()
