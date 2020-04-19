import os

from config_files.configuration_utils import create_configuration
from experiments.reporting.report_repository import ReportRepository
from experiments.slack_client import SlackNotifier
from neat.neat_logger import get_neat_logger
from neat.population_engine import EvolutionEngine
from neat.reporting.reports_pyneat import EvolutionReport
from neat.utils import timeit


config_file = 'iris'
config = create_configuration(filename=f'/{config_file}.json')

LOGS_PATH = f'{os.getcwd()}/'
logger = get_neat_logger(path=LOGS_PATH)

# TODO: better mechanism for override
config.pop_size = 50
config.n_process = 10
config.parallel_evaluation = True
config.n_generations = 300
config.noise = 1.0
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
    for retry in range(2):

        print('Another Try')
        total += 1

        report = EvolutionReport(report_repository=report_repository,
                                 algorithm_version=ALGORITHM_VERSION,
                                 dataset=DATASET,
                                 correlation_id=CORRELATION_ID)
        print(report.report.execution_id)
        evolution_engine = EvolutionEngine(report=report, notifier=notifier)
        evolution_engine.run()

    print(f'It failed {failed} times out of {total}')


main()
