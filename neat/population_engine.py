import math
import random
from itertools import count
import numpy as np
from neat.configuration import get_configuration
from neat.evaluation import EvaluationStochasticEngine
from neat.gene import ConnectionGene, NodeGene
from neat.genome import Genome
from neat.species import SpeciationEngine
from neat.stagnation import Stagnation
from neat.utils import timeit


class EvolutionEngine:

    def __init__(self):
        self.population_engine = PopulationEngine(stagnation_engine=Stagnation())
        self.speciation_engine = SpeciationEngine()
        self.evaluation_engine = EvaluationStochasticEngine()
        self.evolution_configuration = get_configuration()
        self.report = EvolutionReport()
        self.n_generations = self.evolution_configuration.n_generations

        self.population = None

    def run(self):
        # initialize population
        self.population = self.population_engine.initialize_population()
        self.speciation_engine.speciate(self.population, generation=0)

        self.population = self.evaluation_engine.evaluate(population=self.population)

        # report
        self.report.report_new_generation(0, self.population)

        for generation in range(1, self.n_generations + 1):
            self._run_generation(generation)

        self.report.generate_final_report()

    @timeit
    def _run_generation(self, generation):
        # create new generation's population
        self.population = self.population_engine.reproduce(species=self.speciation_engine.species,
                                                           pop_size=self.population_engine.pop_size,
                                                           generation=generation)
        # create new species based on new population
        self.speciation_engine.speciate(self.population, generation=generation)

        # evaluate
        self.population = self.evaluation_engine.evaluate(population=self.population)

        # final generation report
        self.report.report_new_generation(generation, self.population)


class EvolutionReport:

    def __init__(self):
        self.generation_metrics = dict()
        self.best_individual = None

    def report_new_generation(self, generation, population):
        best_individual_key = -1
        best_individual_fitness = 1000000
        fitness_all = []
        for key, genome in population.items():
            fitness_all.append(genome.fitness)
            if genome.fitness < best_individual_fitness:
                best_individual_fitness = genome.fitness
                best_individual_key = genome.key

        data = {'best_individual_fitness': best_individual_fitness,
                'best_individual_key': best_individual_key,
                'all_fitness': fitness_all,
                'min': round(min(fitness_all), 3),
                'max': round(max(fitness_all), 3),
                'mean': round(np.mean(fitness_all), 3)}
        print(f'Generation {generation}. Best fitness: {round(max(fitness_all), 3)}. '
              f'Mean fitness: {round(np.mean(fitness_all), 3)}')
        self.generation_metrics[generation] = data

    def generate_final_report(self):
        pass

    def persist(self):
        pass

    def get_best_individual(self):
        return self.best_individual


class PopulationEngine:
    def __init__(self, stagnation_engine: Stagnation):
        self.stagnation_engine = stagnation_engine
        self.crossover = Crossover()
        self.mutation = Mutation()

        self.config = get_configuration()
        self.pop_size = self.config.pop_size
        self.min_species_size = self.config.min_species_size
        self.elitism = self.config.elitism
        self.survival_threshold = self.config.survival_threshold

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

    @staticmethod
    def compute_spawn(adjusted_fitness, previous_sizes, pop_size, min_species_size):
        """Compute the proper number of offspring per species (proportional to fitness)."""
        af_sum = sum(adjusted_fitness)

        spawn_amounts = []
        for af, ps in zip(adjusted_fitness, previous_sizes):
            if af_sum > 0:
                s = max(min_species_size, af / af_sum * pop_size)
            else:
                s = min_species_size

            d = (s - ps) * 0.5
            c = int(round(d))
            spawn = ps
            if abs(c) > 0:
                spawn += c
            elif d > 0:
                spawn += 1
            elif d < 0:
                spawn -= 1

            spawn_amounts.append(spawn)

        # Normalize the spawn amounts so that the next generation is roughly
        # the population size requested by the user.
        total_spawn = sum(spawn_amounts)
        norm = pop_size / total_spawn
        spawn_amounts = [max(min_species_size, int(round(n * norm))) for n in spawn_amounts]

        return spawn_amounts

    def reproduce(self, species, pop_size, generation):
        """
        Handles creation of genomes, either from scratch or by sexual or
        asexual reproduction from parents.
        """
        all_fitnesses = []
        remaining_species = []
        for stag_sid, stag_s, stagnant in self.stagnation_engine.get_stagnant_species(species, generation):
            if stagnant:
                # TODO: log values
                pass
                # self.reporters.species_stagnant(stag_sid, stag_s)
            else:
                all_fitnesses.extend(m.fitness for m in stag_s.members.values())
                remaining_species.append(stag_s)

        # No species left.
        if not remaining_species:
            species.species = {}
            return {}  # was []

        # Find minimum/maximum fitness across the entire population, for use in
        # species adjusted fitness computation.
        min_fitness = min(all_fitnesses)
        max_fitness = max(all_fitnesses)
        # Do not allow the fitness range to be zero, as we divide by it below.
        # TODO: The ``1.0`` below is rather arbitrary, and should be configurable.
        fitness_range = max(1.0, max_fitness - min_fitness)
        for afs in remaining_species:
            # Compute adjusted fitness.
            msf = np.mean([m.fitness for m in afs.members.values()])
            af = (msf - min_fitness) / fitness_range
            afs.adjusted_fitness = af

        adjusted_fitnesses = [s.adjusted_fitness for s in remaining_species]
        avg_adjusted_fitness = np.mean(adjusted_fitnesses)  # type: float
        # TODO: LOG
        # self.reporters.info("Average adjusted fitness: {:.3f}".format(avg_adjusted_fitness))

        # Compute the number of new members for each species in the new generation.
        previous_sizes = [len(s.members) for s in remaining_species]
        min_species_size = self.min_species_size
        # Isn't the effective min_species_size going to be max(min_species_size,
        # self.reproduction_config.elitism)? That would probably produce more accurate tracking
        # of population sizes and relative fitnesses... doing. TODO: document.
        min_species_size = max(min_species_size, self.elitism)
        spawn_amounts = self.compute_spawn(adjusted_fitnesses, previous_sizes,
                                           pop_size, min_species_size)

        new_population = {}
        species = {}
        for spawn, s in zip(spawn_amounts, remaining_species):
            # If elitism is enabled, each species always at least gets to retain its elites.
            spawn = max(spawn, self.elitism)

            assert spawn > 0

            # The species has at least one member for the next generation, so retain it.
            old_members = list(s.members.items())
            s.members = {}
            species[s.key] = s

            # Sort members in order of descending fitness.
            old_members.sort(reverse=True, key=lambda x: x[1].fitness)

            # Transfer elites to new generation.
            if self.elitism > 0:
                for i, m in old_members[:self.elitism]:
                    new_population[i] = m
                    spawn -= 1

            if spawn <= 0:
                continue

            # Only use the survival threshold fraction to use as parents for the next generation.
            repro_cutoff = int(math.ceil(self.survival_threshold *
                                         len(old_members)))
            # Use at least two parents no matter what the threshold fraction result is.
            repro_cutoff = max(repro_cutoff, 2)
            old_members = old_members[:repro_cutoff]

            # Randomly choose parents and produce the number of offspring allotted to the species.
            while spawn > 0:
                spawn -= 1

                parent1_id, parent1 = random.choice(old_members)
                parent2_id, parent2 = random.choice(old_members)

                # Note that if the parents are not distinct, crossover will produce a
                # genetically identical clone of the parent (but with a different ID).
                key = next(self.genome_indexer)
                offspring = self.crossover.get_offspring(offspring_key=key, genome1=parent1, genome2=parent2)
                mutated_offspring = self.mutation.mutate(genome=offspring)

                new_population[key] = mutated_offspring
                self.ancestors[key] = (parent1_id, parent2_id)

        return new_population


class Crossover:

    def __init__(self):
        self.config = get_configuration()

    def get_offspring(self, offspring_key, genome1: Genome, genome2: Genome):
        assert isinstance(genome1.fitness, (int, float))
        assert isinstance(genome2.fitness, (int, float))
        if genome1.fitness > genome2.fitness:
            parent1, parent2 = genome1, genome2
        else:
            parent1, parent2 = genome2, genome1

        offspring = Genome(key=offspring_key)

        # Inherit connection genes
        for key, cg1 in parent1.connection_genes.items():
            cg2 = parent2.connection_genes.get(key)
            if cg2 is None:
                # Excess or disjoint gene: copy from the fittest parent.
                offspring.connection_genes[key] = cg1.copy()
            else:
                # Homologous gene: combine genes from both parents.
                offspring.connection_genes[key] = self._get_connection_crossover(cg1, cg2)

        # Inherit node genes
        for key, ng1 in parent1.node_genes.items():
            ng2 = parent2.node_genes.get(key)
            assert key not in offspring.node_genes
            if ng2 is None:
                # Extra gene: copy from the fittest parent
                offspring.node_genes[key] = ng1.copy()
            else:
                # Homologous gene: combine genes from both parents.
                offspring.node_genes[key] = self._get_node_crossover(ng1, ng2)
        return offspring

    def _get_node_crossover(self, node_1: NodeGene, node_2: NodeGene):
        assert node_1.key == node_2.key
        node_key = node_1.key

        new_node = NodeGene(key=node_key)

        for attribute in new_node.main_attributes:
            if random.random() > 0.5:
                setattr(new_node, attribute, getattr(node_1, attribute))
            else:
                setattr(new_node, attribute, getattr(node_2, attribute))

        for attribute in new_node.other_attributes:
            if random.random() > 0.5:
                setattr(new_node, attribute, getattr(node_1, attribute))
            else:
                setattr(new_node, attribute, getattr(node_2, attribute))
        return new_node

    def _get_connection_crossover(self, connection_1: ConnectionGene, connection_2: ConnectionGene):
        assert connection_1.key == connection_2.key
        connection_key = connection_1.key

        new_connection = ConnectionGene(key=connection_key)

        for attribute in new_connection.main_attributes:
            if random.random() > 0.5:
                setattr(new_connection, attribute, getattr(connection_1, attribute))
            else:
                setattr(new_connection, attribute, getattr(connection_2, attribute))

        for attribute in new_connection.other_attributes:
            if random.random() > 0.5:
                setattr(new_connection, attribute, getattr(connection_1, attribute))
            else:
                setattr(new_connection, attribute, getattr(connection_2, attribute))
        return new_connection


class Mutation:

    def __init__(self):
        self.config = get_configuration()
        self.single_structural_mutation = self.config.single_structural_mutation
        self.mutate_rate = self.config.mutate_rate
        self.mutate_power = self.config.mutate_power
        self.replace_rate = self.config.replace_rate

        # self.node_add_prob = self.config.node_add_prob
        # self.node_delete_prob = self.config.node_delete_prob
        # self.conn_add_prob = self.config.conn_add_prob
        # self.conn_delete_prob = self.config.conn_delete_prob

    def mutate(self, genome: Genome):
        # if self.single_structural_mutation:
        #     div = max(1, (self.node_add_prob + self.node_delete_prob +
        #                   self.conn_add_prob + self.conn_delete_prob))
        #     r = random.random()
        #     if r < (self.node_add_prob/div):
        #         genome = self.mutate_add_node(genome)
        #     elif r < ((self.node_add_prob + self.node_delete_prob)/div):
        #         genome = self.mutate_delete_node(genome)
        #     elif r < ((self.node_add_prob + self.node_delete_prob +
        #                self.conn_add_prob)/div):
        #         genome = self.mutate_add_connection(genome)
        #     elif r < ((self.node_add_prob + self.node_delete_prob +
        #                self.conn_add_prob + self.conn_delete_prob)/div):
        #         genome = self.mutate_delete_connection(genome)
        # else:
        #     if random.random() < self.node_add_prob:
        #         genome = self.mutate_add_node(genome)
        #
        #     if random.random() < self.node_delete_prob:
        #         genome = self.mutate_delete_node(genome)
        #
        #     if random.random() < self.conn_add_prob:
        #         genome = self.mutate_add_connection(genome)
        #
        #     if random.random() < self.conn_delete_prob:
        #         genome = self.mutate_delete_connection(genome)

        # Mutate connection genes.
        for key in genome.connection_genes.keys():
            genome.connection_genes[key] = self.mutate_connection(connection=genome.connection_genes[key])

        # Mutate node genes (bias, response, etc.).
        for key in genome.node_genes.keys():
            genome.node_genes[key] = self.mutate_node(node=genome.node_genes[key])
        return genome

    def mutate_add_node(self, genome):
        raise
        return genome

    def mutate_delete_node(self, genome):
        pass

    def mutate_add_connection(self, genome):
        pass

    def mutate_delete_connection(self, genome: Genome):
        pass

    def mutate_connection(self, connection: ConnectionGene):
        for attribute in connection.main_attributes:
            attribute_value = getattr(connection, attribute)
            mutated_value = self._mutate_float(value=attribute_value, name=attribute)
            setattr(connection, attribute, mutated_value)

        # for attribute in connection.other_attributes:
        #     attribute_value = getattr(connection, attribute)
        #     mutated_value = self._mutate_float(value=attribute_value, name=attribute)
        #     setattr(connection, attribute, mutated_value)
        return connection

    def mutate_node(self, node: NodeGene):
        for attribute in node.main_attributes:
            attribute_value = getattr(node, attribute)
            mutated_value = self._mutate_float(value=attribute_value, name=attribute)
            setattr(node, attribute, mutated_value)

        # for attribute in node.other_attributes:
        #     attribute_value = getattr(node, attribute)
        #     mutated_value = self._mutate_float(value=attribute_value, name=attribute)
        #     setattr(node, attribute, mutated_value)
        return node

    def _mutate_float(self, value, name: str):
        r = random.random()
        if r < self.mutate_rate:
            mutated_value = value + random.gauss(0.0, self.mutate_power)
            mutated_value = self._clip(value=mutated_value,
                                       name=name)
            return mutated_value

        if r < self.replace_rate + self.mutate_rate:
            return self._init_float(name=name)

        return value

    def _clip(self, value, name):
        min_ = getattr(self.config, f'{name}_min_value')
        max_ = getattr(self.config, f'{name}_max_value')
        return np.clip(value, a_min=min_, a_max=max_)

    def _init_float(self, name):
        mean = getattr(self.config, f'{name}_init_mean')
        std = getattr(self.config, f'{name}_init_std')

        value = np.random.normal(loc=mean, scale=std)
        return self._clip(value=value, name=name)
