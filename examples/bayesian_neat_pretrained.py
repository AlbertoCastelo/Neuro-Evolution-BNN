import os

import torch

from config_files.configuration_utils import create_configuration
from deep_learning.standard.runner import StandardDLRunner
from experiments.reporting.report_repository import ReportRepository
from experiments.slack_client import SlackNotifier
from neat.evaluation.evaluate_simple import evaluate_genome
from neat.evaluation.utils import get_dataset
from neat.loss.vi_loss import get_loss
from neat.neat_logger import get_neat_logger
from neat.population_engine import EvolutionEngine
from neat.reporting.reports_pyneat import EvolutionReport
from neat.representation_mapping.network_to_genome.standard_feed_forward_to_genome import \
    get_genome_from_standard_network
from neat.utils import timeit


config_file = 'mnist_downsampled'
config = create_configuration(filename=f'/{config_file}.json')

LOGS_PATH = f'{os.getcwd()}/'
logger = get_neat_logger(path=LOGS_PATH)

# TODO: better mechanism for override
config.pop_size = 50
config.n_process = 10
config.parallel_evaluation = False
config.n_generations = 300
# config.n_samples = 50
# config.fix_std = False

config.n_samples = 1
config.fix_std = True

ALGORITHM_VERSION = 'bayes-neat'
DATASET = config_file
CORRELATION_ID = 'tests'

@timeit
def main():
    report_repository = ReportRepository.create(project='neuro-evolution', logs_path=LOGS_PATH)
    notifier = SlackNotifier.create(channel='batch-jobs')
    failed = 0
    total = 0
    for retry in range(1):
        logger.info(f'Pretraining Network')
        standard_runner = StandardDLRunner(config=config, n_epochs=100)
        standard_runner.run()
        # TODO: save pretrained results to Report

        config.initial_genome_filename = save_genome(network=standard_runner.evaluator.network,
                                                     dataset=config.dataset)

        config.beta = 0.000005
        config.architecture_mutation_power = 5
        config.node_add_prob = 0.0
        config.node_delete_prob = 0.0
        config.connection_add_prob = 0.0
        config.connection_delete_prob = 1.0
        config.mutate_power = 0.1
        print('Another Try')
        total += 1

        report = EvolutionReport(report_repository=report_repository,
                                 algorithm_version=ALGORITHM_VERSION,
                                 dataset=DATASET,
                                 correlation_id=CORRELATION_ID)
        print(report.report.execution_id)
        evolution_engine = EvolutionEngine(report=report, notifier=notifier)
        evolution_engine.run()

        #
        # dataset = get_dataset(config.dataset, train_percentage=config.train_percentage, testing=False,
        #                       random_state=config.dataset_random_state)
        # loss = get_loss(problem_type=config.problem_type)
        # x, y_true, y_pred, loss_value = evaluate_genome(genome=genome,
        #                                                 dataset=dataset,
        #                                                 loss=loss,
        #                                                 problem_type=config.problem_type,
        #                                                 beta_type=config.beta_type,
        #                                                 batch_size=config.batch_size,
        #                                                 n_samples=config.n_samples,
        #                                                 is_gpu=config.is_gpu,
        #                                                 is_testing=True,
        #                                                 return_all=True)
        # y_pred = torch.argmax(y_pred, dim=1)
        # from sklearn.metrics import confusion_matrix, accuracy_score
        # # print(f'Loss: {loss_value}')
        # confusion_m = confusion_matrix(y_true, y_pred)
        # acc = accuracy_score(y_true, y_pred) * 100
        #
        # print('Confusion Matrix:')
        # print(confusion_m)
        # print(f'Accuracy: {acc} %')



    print(f'It failed {failed} times out of {total}')


def save_genome(network, dataset):
    filename = ''.join([os.path.dirname(os.path.realpath(__file__)), f'/genomes/genome-{dataset}.json'])
    genome = get_genome_from_standard_network(network=network)
    genome.save_genome(filename)
    return filename

main()
