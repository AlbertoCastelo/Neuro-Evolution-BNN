import math
import random
import time
from itertools import count
import numpy as np

from experiments.logger import logger
from experiments.slack_client import Notifier
from neat.configuration import get_configuration
from neat.evaluation.evaluation_engine import EvaluationStochasticEngine
from neat.evaluation.utils import get_dataset
from neat.evolution_operators.backprop_mutation import BackPropMutation, BACKPROP_MUTATION
from neat.evolution_operators.crossover import Crossover
from neat.evolution_operators.mutation import RandomMutation, ArchitectureMutation, RANDOM_MUTATION
from neat.finetuning import FineTuner
from neat.genome import Genome
from neat.reporting.reports_pyneat import EvolutionReport
from neat.species import SpeciationEngine, FixSpeciationEngine
from neat.stagnation import Stagnation
from neat.utils import timeit

TIMEOUT_SECONDS = 3600
N_EPOCHS = 5
LR = 0.05
WEIGHT_DECAY = 0.0005


class EvolutionEngine:

    def __init__(self, report: EvolutionReport, notifier: Notifier, is_cuda: bool = False):
        self.report = report
        self.notifier = notifier
        self.is_cuda = is_cuda

        self.population_engine = PopulationEngine(stagnation_engine=Stagnation())
        # self.speciation_engine = SpeciationEngine()
        self.speciation_engine = FixSpeciationEngine()
        self.evaluation_engine = EvaluationStochasticEngine()
        self.evolution_configuration = get_configuration()

        self.n_generations = self.evolution_configuration.n_generations

        self.population = None

        self.start_time = time.perf_counter()

    @timeit
    def run(self):
        logger.info('Started evolutionary process')
        end_condition = 'normal'
        # try:
        # initialize population
        self.population = self.population_engine.initialize_population()
        self.speciation_engine.speciate(self.population, generation=0)

        self.population = self.evaluation_engine.evaluate(population=self.population)

        # report
        self.report.report_new_generation(generation=0,
                                          population=self.population,
                                          species=self.speciation_engine.species)

        for generation in range(1, self.n_generations + 1):
            self._run_generation(generation)

            elapsed = time.perf_counter() - self.start_time
            if elapsed > TIMEOUT_SECONDS:
                end_condition = 'timeout'
                break

        if self.evolution_configuration.is_fine_tuning:
            fine_tuner = FineTuner(species=self.speciation_engine.species, config=self.evolution_configuration,
                                   is_cuda=self.is_cuda, only_best=False)
            fine_tuner.run()
            best_genomes = fine_tuner.species_best_genome
            # best_genomes = self.evaluation_engine.evaluate(population=best_genomes)
            self.report.report_fine_tuning(best_genomes)

        self.evaluation_engine.close()

        self.report.generate_final_report(end_condition=end_condition)\
                   .persist_report()
        self.report.persist_logs()
        # self.notifier.send(str(self.report.get_best_individual()))
        self._send_final_message()
        logger.info('Finished evolutionary process')

    @timeit
    def _run_generation(self, generation):
        # create new generation's population
        self.population = self.population_engine.reproduce(species=self.speciation_engine.species,
                                                           pop_size=self.population_engine.pop_size,
                                                           generation=generation)

        # evaluate
        self.population = self.evaluation_engine.evaluate(population=self.population)

        # create new species based on new population
        self.speciation_engine.speciate(self.population, generation=generation)

        # generation report
        self.report.report_new_generation(generation=generation,
                                          population=self.population,
                                          species=self.speciation_engine.species)

        # schedule parameters
        if generation == self.evolution_configuration.generation_fix_architecture:
            logger.info('Fixing Architecture')
            self.evolution_configuration.fix_architecture = True

    def _send_final_message(self):
        self.notifier.send(f'-----------------------------------------------------------------\n'
                           f'Dataset: {self.report.dataset}. \nCorrelation id: {self.report.correlation_id}.\n'
                           f'Execution id: {self.report.report.execution_id}')
        self.notifier.send(f'Best Fitness: {self.report.best_individual.fitness}')
        if self.report.generic_text is not None:
            self.notifier.send(str(self.report.generic_text))
        self.notifier.send(str(self.report.metrics_best))


class PopulationEngine:
    def __init__(self, stagnation_engine: Stagnation):
        self.stagnation_engine = stagnation_engine
        self.crossover = Crossover()
        self.architecture_mutation = ArchitectureMutation()

        self.config = get_configuration()
        self.mutation_type = self.config.mutation_type
        self.params_mutation = self._initialize_params_mutation(mutation_type=self.mutation_type, config=self.config)
        self.pop_size = self.config.pop_size
        self.min_species_size = self.config.min_species_size
        self.elitism = self.config.elitism
        self.survival_threshold = self.config.survival_threshold

        self.genome_indexer = count(1)
        self.ancestors = {}

    @timeit
    def initialize_population(self):
        population = {}
        for i in range(self.pop_size):
            key = next(self.genome_indexer)
            if self.config.initial_genome_filename is None:
                genome = Genome(key=key)
                genome.create_random_genome()
            else:
                filename = self.config.initial_genome_filename
                genome = Genome.create_from_file(filename=filename, key=key)

            population[key] = genome
            self.ancestors[key] = tuple()
        return population

    @staticmethod
    @timeit
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

    @timeit
    def reproduce(self, species, pop_size, generation):
        """
        Disclaimer: this is taken from Python-NEAT
        Handles creation of genomes, either from scratch or by sexual or
        asexual reproduction from parents.
        """
        all_fitnesses = []
        remaining_species = []
        for stag_sid, stag_s, stagnant in self.stagnation_engine.get_stagnant_species(species, generation):
            if stagnant:
                logger.debug(f'Stagnant specie: {stag_sid} - {stag_s}')
            else:
                all_fitnesses.extend(m.fitness for m in stag_s.members.values())
                remaining_species.append(stag_s)

        # No species left.
        if not remaining_species:
            species.species = {}
            raise ValueError('No species left. Reproduction failed...')
            # return {}  # was []

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

        new_population = self._create_new_population(remaining_species, spawn_amounts)

        return new_population

    @timeit
    def _create_new_population(self, remaining_species, spawn_amounts):
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
                mutated_offspring = self.architecture_mutation.mutate(genome=offspring)
                mutated_offspring = self.params_mutation.mutate(genome=mutated_offspring)

                new_population[key] = mutated_offspring
                self.ancestors[key] = (parent1_id, parent2_id)
        return new_population

    def _initialize_params_mutation(self, mutation_type, config):
        if mutation_type == RANDOM_MUTATION:
            return RandomMutation()
        elif mutation_type == BACKPROP_MUTATION:
            dataset = get_dataset(dataset=config.dataset, train_percentage=config.train_percentage, testing=False,
                                  random_state=config.dataset_random_state, noise=config.noise)
            return BackPropMutation(dataset=dataset, n_samples=config.n_samples, problem_type=config.problem_type,
                                    beta=config.beta, n_epochs=N_EPOCHS, weight_decay=WEIGHT_DECAY, lr=LR)
        else:
            raise ValueError('Incorrect Mutation Type')
