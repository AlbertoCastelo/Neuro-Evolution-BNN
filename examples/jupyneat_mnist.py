import os

from config_files.configuration_utils import create_configuration
from experiments.reporting.report_repository import ReportRepository
from experiments.slack_client import SlackNotifier
from neat.evaluation_engine import JupyNeatFSEvaluationEngine
from neat.neat_logger import get_neat_logger
from neat.reporting.reports_jupyneat import EvolutionReportJupyNeat
from neat.utils import timeit


# config_file = 'mnist'
DATASET = 'mnist_downsampled'

config = create_configuration(filename=f'/{DATASET}.json')

LOGS_PATH = f'{os.getcwd()}/'
logger = get_neat_logger(path=LOGS_PATH)

# TODO: better mechanism for override
config.n_generations = 20
config.pop_size = 150
config.n_samples = 20

config.max_stagnation = 30
config.node_add_prob = 0.5


ALGORITHM_VERSION = 'bayes-neat'
# DATASET = 'mnist'

CORRELATION_ID = 'test'


@timeit
def main():
    report_repository = ReportRepository.create(project='neuro-evolution', logs_path=LOGS_PATH)
    notifier = SlackNotifier.create(channel='batch-jobs')

    report = EvolutionReportJupyNeat(report_repository=report_repository,
                                     algorithm_version=ALGORITHM_VERSION,
                                     dataset=DATASET,
                                     correlation_id=CORRELATION_ID,
                                     configuration=config)
    print(report.report.execution_id)

    config.experiment = CORRELATION_ID
    config.dataset = DATASET
    config.execution = report.report.execution_id
    # execute scenario
    evaluation_engine = JupyNeatFSEvaluationEngine.create(report=report, notifier=notifier)
    evaluation_engine.run()


main()

