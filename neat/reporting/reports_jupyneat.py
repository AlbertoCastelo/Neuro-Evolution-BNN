import numpy as np

from experiments.logger import logger
from experiments.reporting.report import BaseReport
from experiments.reporting.report_repository import ReportRepository
from neat.genome import Genome
from neat.utils import timeit


class EvolutionReportJupyNeat:

    @staticmethod
    def create(report_repository: ReportRepository, algorithm_version, dataset, correlation_id, configuration):
        return EvolutionReportJupyNeat(report_repository=report_repository, algorithm_version=algorithm_version,
                                       dataset=dataset, correlation_id=correlation_id, configuration=configuration)

    def __init__(self, report_repository: ReportRepository, algorithm_version, dataset, correlation_id, configuration):
        self.report_repository = report_repository
        self.algorithm_version = algorithm_version
        self.dataset = dataset
        self.correlation_id = correlation_id
        self.configuration = configuration

        self.report = BaseReport(correlation_id)
        self.report.add('algorithm_version', algorithm_version)
        self.report.add('dataset', dataset)
        self.report.add('config', self.configuration.to_dict())

        self.generation_metrics = dict()
        self.best_individual = None

    @timeit
    def report_new_generation(self, generation: int, population: dict, species: dict):
        generation_report = GenerationReport.create(population=population, generation=generation, species=species).run()
        generation_data = generation_report.generation_data

        is_best_updated = self._update_best(generation_report=generation_report, population=population)

        self.generation_metrics[generation] = generation_data

        if is_best_updated:
            self._generate_report(end_condition='checkpoint')
            self.persist_report()

    def _update_best(self, generation_report, population):
        if self.best_individual is None or self.best_individual['fitness'] < generation_report.best_individual_fitness:
            self.best_individual = population.get(generation_report.best_individual_key)
            logger.info(f'''    New best individual ({self.best_individual['key']}) found '''
                        f'''with fitness {round(self.best_individual['fitness'], 3)}''')
            logger.debug(f'''         best individual has {len(self.best_individual['nodes'])} Nodes '''
                         f'''and {len(self.best_individual['connections'])} Connections''')
            return True
        return False

    def generate_final_report(self, end_condition):
        self._generate_report(end_condition)
        return self

    def _generate_report(self, end_condition='normal'):

        self.report.add_data(name='generation_metrics', value=self.generation_metrics)
        self.report.add_data(name='best_individual', value=self.get_best_individual())
        # self.report.add_data(name='best_individual_graph', value=self.get_best_individual().get_graph())
        self.report.add_data(name='best_individual_fitness', value=self.get_best_individual()['fitness'])
        self.report.add_data(name='end_condition', value=end_condition)
        self.report.set_finish_time()

    def persist_report(self):
        self.report_repository.set_report(report=self.report)
        return self

    def persist_logs(self):
        self.report_repository.persist_logs(algorithm_version=self.algorithm_version, dataset=self.dataset,
                                            correlation_id=self.correlation_id, execution_id=self.report.execution_id)

    def get_best_individual(self) -> Genome:
        return self.best_individual


def calculate_number_of_parameters(genome: dict):
    n_weight_parameters = 2 * len(genome['connections'])
    n_bias_parameters = 2 * len(genome['nodes'])
    return n_weight_parameters + n_bias_parameters


class GenerationReport:
    @staticmethod
    def create(population: dict, species: dict, generation: int):
        return GenerationReport(population=population,
                                generation=generation,
                                species=species)

    def __init__(self, population: dict, species: dict, generation: int):
        self.population = population
        self.species = species
        self.generation = generation
        self.generation_data = {}

    def run(self):
        self._prepare_population_report()
        if self.species is not None:
            self._prepare_species_report()
        return self

    def _prepare_population_report(self):
        self.best_individual_key = -1
        self.best_individual_fitness = -1000000
        fitness_all = []
        all_n_parameters = []
        for key, genome in self.population.items():
            fitness_all.append(genome['fitness'])

            all_n_parameters.append(calculate_number_of_parameters(genome))
            if genome['fitness'] > self.best_individual_fitness:
                self.best_individual_key = genome['key']
                self.best_individual_fitness = genome['fitness']



        self.generation_data['best_individual_fitness'] = self.best_individual_fitness
        self.generation_data['best_individual_key'] = self.best_individual_key
        # self.generation_data['best_individual_graph'] = self.population.get(self.best_individual_key).get_graph()
        self.generation_data['all_fitness'] = fitness_all
        self.generation_data['min_fitness'] = round(min(fitness_all), 3)
        self.generation_data['max_fitness'] = round(max(fitness_all), 3)
        self.generation_data['mean_fitness'] = round(np.mean(fitness_all), 3)
        self.generation_data['std_fitness'] = round(np.std(fitness_all), 3)

        # best_n_parameters = self.population.get(self.best_individual_key).calculate_number_of_parameters()
        best_n_parameters = calculate_number_of_parameters(self.population.get(self.best_individual_key))

        logger.info(f'Generation {self.generation}. Best fitness: {round(max(fitness_all), 3)}. '
                    f'N-Parameters Best: {best_n_parameters}')
        logger.info(f'                         Mean fitness: {round(np.mean(fitness_all), 3)}. '
                    f'Mean N-Parameters: {round(np.mean(all_n_parameters), 3)}')
        logger.info(f'                         N-Species: {len(self.species)}')

    def _prepare_species_report(self):
        self.generation_data['n_species'] = len(self.species)
        self.best_specie_key = -1
        self.best_specie_fitness = -10000000
        fitness_all = []
        genomes_fitness_all = {}
        genomes_per_specie = {}
        for key, specie in self.species.items():
            fitness_all.append(specie.fitness)
            genomes_per_specie[key] = list(specie.members.keys())
            genomes_fitness_all[key] = [genome.fitness for genome in list(specie.members.values())]
        self.generation_data['genomes_per_specie'] = genomes_per_specie
        self.generation_data['genomes_fitness_per_specie'] = genomes_fitness_all
