from experiments.logger import get_logger
from experiments.slack_client import SlackNotifier
from neat.population_engine import EvolutionEngine
from neat.reports import EvolutionReport
from neat.utils import timeit
from tests.config_files.config_files import create_configuration

config_file = 'classification-miso'
config = create_configuration(filename=f'/{config_file}.json')
logger = get_logger(path='./')

# TODO: better mechanism for override
config.n_generations = 50
config.pop_size = 20


@timeit
def main():
    report = EvolutionReport(experiment_name=f'{config_file}_v3')
    evolution_engine = EvolutionEngine(report=report)
    evolution_engine.run()
    return evolution_engine

evolution_engine = main()

best_individual = evolution_engine.report.get_best_individual()
# report to slack
notifier = SlackNotifier.create(channel='batch-jobs')
notifier.send(message=f'{str(best_individual)}')
