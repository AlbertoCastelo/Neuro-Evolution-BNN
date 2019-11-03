from experiments.logger import get_logger
from experiments.reporting.report_repository import ReportRepository
from experiments.slack_client import SlackNotifier
from neat.population_engine import EvolutionEngine
from neat.reports import EvolutionReport
from neat.utils import timeit
from tests.config_files.config_files import create_configuration

config_file = 'classification-miso'
config = create_configuration(filename=f'/{config_file}.json')
logger = get_logger(path='./')

# TODO: better mechanism for override
config.n_generations = 500
config.pop_size = 25

ALGORITHM_VERSION = 'bayes-neat'
DATASET = 'toy-classification'
CORRELATION_ID = 'parameters_grid'


@timeit
def main():

    report_repository = ReportRepository.create(project='neuro-evolution')
    notifier = SlackNotifier.create(channel='batch-jobs')
    for pop_size in range(25, 201, 25):
        for retry in range(5):

            report = EvolutionReport(report_repository=report_repository,
                                     algorithm_version=ALGORITHM_VERSION,
                                     dataset=DATASET,
                                     correlation_id=CORRELATION_ID)

            evolution_engine = EvolutionEngine(report=report, notifier=notifier)
            evolution_engine.run()
    # return evolution_engine


main()
# evolution_engine = main()
#
# best_individual = evolution_engine.report.get_best_individual()
# # report to slack
# notifier.send(message=f'{str(best_individual)}')
