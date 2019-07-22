from neat.population_engine import EvolutionEngine
from neat.configuration import get_configuration


def run():
    # initialize configuration
    config = get_configuration()

    # setup evolution
    evolution = EvolutionEngine()
    evolution.run()


if __name__ == '__main__':
    run()
