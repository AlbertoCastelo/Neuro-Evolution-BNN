import os
from experiments.reporting.report_repository import ReportRepository
from experiments.slack_client import SlackNotifier
from neat.evaluation_engine import JupyNeatEvaluationEngine
from neat.neat_logger import get_neat_logger
from neat.reports import EvolutionReport
from neat.utils import timeit
from tests.config_files.config_files import create_configuration


config_file = 'mnist_binary'
config = create_configuration(filename=f'/{config_file}.json')

LOGS_PATH = f'{os.getcwd()}/'
logger = get_neat_logger(path=LOGS_PATH)

# TODO: better mechanism for override
config.n_generations = 3
config.pop_size = 200
config.n_samples = 20

config.max_stagnation = 30
config.node_add_prob = 0.5

ALGORITHM_VERSION = 'bayes-neat'
DATASET = 'mnist_binary'
CORRELATION_ID = 'test'

@timeit
def main():
    report_repository = ReportRepository.create(project='neuro-evolution', logs_path=LOGS_PATH)
    notifier = SlackNotifier.create(channel='batch-jobs')

    report = EvolutionReport(report_repository=report_repository,
                             algorithm_version=ALGORITHM_VERSION,
                             dataset=DATASET,
                             correlation_id=CORRELATION_ID)
    print(report.report.execution_id)

    config.experiment = CORRELATION_ID
    config.dataset = DATASET
    config.execution = report.report.execution_id
    # execute scenario
    evaluation_engine = JupyNeatEvaluationEngine.create(report=report, notifier=notifier)
    evaluation_engine.run()


main()

