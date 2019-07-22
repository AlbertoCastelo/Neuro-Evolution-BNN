from itertools import count

from neat.configuration import get_configuration
from neat.evaluation import EvaluationEngine
from neat.genome import Genome


class EvolutionEngine:

    def __init__(self):
        self.population_engine = PopulationEngine()
        self.evaluation_engine = EvaluationEngine()
        self.evolution_configuration = get_configuration()
        self.n_generations = self.evolution_configuration.n_generations

    def run(self):
        # initialize population
        population = self.population_engine.initialize_population()

        population = self.evaluation_engine.evaluate(population=population)

        for generation in range(self.n_generations):
            pass


class PopulationEngine:
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
