import os

import torch

from config_files.configuration_utils import create_configuration
from experiments.reporting.report_repository import ReportRepository
from experiments.slack_client import SlackNotifier
from neat.neat_logger import get_neat_logger
from neat.population_engine import EvolutionEngine
from neat.reporting.reports_pyneat import EvolutionReport
from neat.utils import timeit, get_slack_channel

# config_file = 'classification-miso'
# config_file = 'iris'
# config_file = 'titanic'
dataset_name = 'mnist_downsampled'
# config_file = 'wine'
# config_file = 'breast_cancer'
config = create_configuration(filename=f'/{dataset_name}.json')

config.n_generations = 150
config.epochs_fine_tuning = 4000

config.n_input = 64
config.is_discrete = False
config.label_noise = 0.0
config.is_initial_fully_connected = True
config.initial_nodes_sample = 32
config.n_initial_hidden_neurons = 0

LOGS_PATH = f'{os.getcwd()}/'
logger = get_neat_logger(path=LOGS_PATH)

# TODO: better mechanism for override
config.pop_size = 50
config.n_processes = 14
config.parallel_evaluation = True
config.train_percentage = 0.75
config.noise = 0.0
config.label_noise = 0.0

config.bias_mean_max_value = 10.0
config.bias_mean_min_value = -10.0
config.bias_std_max_value = 2.0
config.bias_std_min_value = 0.000001

config.weight_mean_max_value = 10.0
config.weight_mean_min_value = -10.0
config.weight_std_max_value = 2.0
config.weight_std_min_value = 0.000001

config.node_delete_prob = 0.0
config.connection_delete_prob = 0.0
config.node_add_prob = 1.0
config.connection_add_prob = 1.0

n_species = 5
architecture_mutation_power = 1


config.n_samples = 50
config.fix_std = False

# config.beta = 0.0
# config.n_initial_hidden_neurons = 0
config.fix_architecture = False
config.beta = 0.005

ALGORITHM_VERSION = 'bayes-neat'
DATASET = dataset_name
CORRELATION_ID = f'tests_new_network_{DATASET}'

is_cuda = True
if is_cuda:
    is_available = torch.cuda.is_available()
    torch.cuda.set_device(0)
    # is_cuda = True
    print('Using GPU for FineTuning')


@timeit
def main():
    report_repository = ReportRepository.create(project='neuro-evolution', logs_path=LOGS_PATH)
    notifier = SlackNotifier.create(channel=get_slack_channel(dataset_name=dataset_name))

    failed = 0
    total = 0
    for retry in range(1):

        print('Another Try')
        total += 1

        report = EvolutionReport(report_repository=report_repository,
                                 algorithm_version=ALGORITHM_VERSION,
                                 dataset=DATASET,
                                 correlation_id=CORRELATION_ID)
        print(report.report.execution_id)
        evolution_engine = EvolutionEngine(report=report, notifier=notifier, is_cuda=is_cuda)
        evolution_engine.run()

    print(f'It failed {failed} times out of {total}')


main()
