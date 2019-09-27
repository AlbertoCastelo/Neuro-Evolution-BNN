from neat.population_engine import EvolutionEngine
from neat.utils import timeit
from tests.config_files.config_files import create_configuration

config_file = '/classification-miso.json'
config = create_configuration(filename=config_file)

@timeit
def main():
    evolution_engine = EvolutionEngine()
    evolution_engine.run()
    return evolution_engine

evolution_engine = main()

best_individual = evolution_engine.report.get_best_individual()

