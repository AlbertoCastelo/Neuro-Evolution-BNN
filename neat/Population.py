from itertools import count

from neat.configuration import get_configuration
from neat.genome import Genome


class Evolution:

    def __init__(self):
        self.population = Population()
        self.evolution_configuration = get_configuration()
        self.n_generations = self.evolution_configuration.n_generations

    def run(self):
        # initialize population
        pop = self.population.initialize_population()

        pop = self.evalu

        for generation in range(self.n_generations):
            pass



class Population:
    def __init__(self):
        self.config = get_configuration()
        self.pop_size = self.config.pop_size

        self.genome_indexer = count(1)
        self.ancestors = {}

    def initialize_population(self):
        population = {}
        for i in range(self.pop_size):
            key = next(self.genome_indexer)
            genome = Genome(key=key)
            genome.create_random_genome()

            population[key] = genome
            self.ancestors[key] = tuple()
        return population

