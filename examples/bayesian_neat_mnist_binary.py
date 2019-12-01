import os
from experiments.reporting.report_repository import ReportRepository
from experiments.slack_client import SlackNotifier
from neat.neat_logger import get_neat_logger
from neat.population_engine import EvolutionEngine
from neat.reports import EvolutionReport
from neat.utils import timeit
from tests.config_files.config_files import create_configuration


config_file = 'mnist_binary'
config = create_configuration(filename=f'/{config_file}.json')

LOGS_PATH = f'{os.getcwd()}/'
logger = get_neat_logger(path=LOGS_PATH)

# TODO: better mechanism for override
config.n_generations = 1000
config.pop_size = 20
config.n_samples = 20

config.max_stagnation = 30
config.node_add_prob = 0.5


ALGORITHM_VERSION = 'bayes-neat'
DATASET = 'mnist_binary'
# CORRELATION_ID = 'parameters_grid'
CORRELATION_ID = 'tests'


@timeit
def main():
    report_repository = ReportRepository.create(project='neuro-evolution', logs_path=LOGS_PATH)
    notifier = SlackNotifier.create(channel='batch-jobs')
    failed = 0
    total = 0
    for pop_size in range(100, 201, 50):
        # for n_samples in [20, 50, 100]:
        for retry in range(2):
            # config.n_samples = n_samples
            config.pop_size = pop_size
            total += 1
            try:
                report = EvolutionReport(report_repository=report_repository,
                                         algorithm_version=ALGORITHM_VERSION,
                                         dataset=DATASET,
                                         correlation_id=CORRELATION_ID)
                print(report.report.execution_id)
                evolution_engine = EvolutionEngine(report=report, notifier=notifier)
                evolution_engine.run()
            except Exception as e:
                print(e)
                notifier.send(e)
                logger.error(e)
                failed += 1
    print(f'It failed {failed} times out of {total}')

    # return evolution_engine


main()
# evolution_engine = main()
#
# best_individual = evolution_engine.report.get_best_individual()
# # report to slack
# notifier.send(message=f'{str(best_individual)}')
