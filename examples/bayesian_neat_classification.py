from neat.population_engine import EvolutionEngine
from neat.reports import EvolutionReport
from neat.utils import timeit
from tests.config_files.config_files import create_configuration

config_file = '/classification-miso.json'
config = create_configuration(filename=config_file)

@timeit
def main():
    report = EvolutionReport(experiment_name='fixed_std')
    evolution_engine = EvolutionEngine(report=report)
    evolution_engine.run()
    return evolution_engine

evolution_engine = main()

best_individual = evolution_engine.report.get_best_individual()

