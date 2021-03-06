import numpy as np
import torch

from experiments.file_utils import write_json_file_from_dict
from experiments.logger import logger
from experiments.reporting.report import BaseReport
from experiments.reporting.report_repository import ReportRepository
from neat.configuration import get_configuration, ConfigError
from neat.evaluation.evaluate_simple import evaluate_genome
from neat.evaluation.utils import get_dataset
from neat.genome import Genome
from neat.loss.vi_loss import get_loss
from neat.utils import timeit


class EvolutionReport:

    @staticmethod
    def create(report_repository: ReportRepository, algorithm_version, dataset, correlation_id=None):
        return EvolutionReport(report_repository=report_repository, algorithm_version=algorithm_version,
                               dataset=dataset, correlation_id=correlation_id)

    def __init__(self, report_repository: ReportRepository, algorithm_version, dataset, correlation_id=None):
        self.report_repository = report_repository
        self.algorithm_version = algorithm_version
        self.dataset = dataset
        self.correlation_id = correlation_id

        self.report = BaseReport(correlation_id)
        self.report.add('algorithm_version', algorithm_version)
        self.report.add('dataset', dataset)

        self.generation_metrics = dict()
        self.fine_tuning = {}
        self.best_individual = None
        self.metrics_best = {}
        self.generic_text = None

    @timeit
    def report_new_generation(self, generation: int, population: dict, species: dict):
        generation_report = GenerationReport.create(population=population, generation=generation, species=species).run()
        generation_data = generation_report.generation_data

        is_best_updated = self._update_best(
            potential_best_individual=population.get(generation_data['best_individual_key']))

        self.generation_metrics[generation] = generation_data

        if is_best_updated:
            self._generate_report(end_condition='checkpoint')
            self.persist_report()

            # show metrics for new one
            self.show_metrics_best()

    def report_fine_tuning(self, species_best_genome):
        self.fine_tuning['best_fitness_before_fine_tuning'] = self.best_individual.fitness
        self.fine_tuning['best_genome_before_fine_tuning'] = self.best_individual.to_dict()
        best_genome_per_specie = {}
        best_genome_specie_key = -1
        best_genome_fitness = -1000000
        for specie_key, best_genome in species_best_genome.items():
            best_genome_per_specie[specie_key] = best_genome.fitness
            if best_genome.fitness > best_genome_fitness:
                best_genome_fitness = best_genome.fitness
                best_genome_specie_key = specie_key
        best_genome = species_best_genome.get(best_genome_specie_key)

        self.fine_tuning['best_genome_per_specie'] = best_genome_per_specie
        is_best_updated = self._update_best(potential_best_individual=best_genome)

        if is_best_updated:
            # show metrics for new one
            self.show_metrics_best()

        print(f'Best Fitness after fine-tuning: {best_genome.fitness}')

    def show_metrics_best(self):
        # only for classification!!
        config = get_configuration()
        if config.problem_type == 'regression':
            # loss = nn.MSELoss()
            pass
        elif config.problem_type == 'classification':
            self._show_classification_metrics(config)
        else:
            raise ConfigError(f'Problem Type is incorrect: {config.problem_type}')

    def _show_classification_metrics(self, config):
        dataset = get_dataset(config.dataset,
                              train_percentage=config.train_percentage,
                              testing=False,
                              random_state=config.dataset_random_state,
                              noise=config.noise,
                              label_noise=config.label_noise)
        loss = get_loss(problem_type=config.problem_type)
        x, y_true, y_pred, loss_value = evaluate_genome(genome=self.best_individual,
                                                        dataset=dataset,
                                                        loss=loss,
                                                        problem_type=config.problem_type,
                                                        beta_type=config.beta_type,
                                                        batch_size=config.batch_size,
                                                        n_samples=config.n_samples,
                                                        is_gpu=config.is_gpu,
                                                        is_testing=True,
                                                        return_all=True)
        y_pred = torch.argmax(y_pred, dim=1)
        from sklearn.metrics import confusion_matrix, accuracy_score
        # print(f'Loss: {loss_value}')
        confusion_m = confusion_matrix(y_true, y_pred)
        acc = accuracy_score(y_true, y_pred) * 100
        self.metrics_best['confusion_matrix'] = confusion_m
        self.metrics_best['accuracy'] = acc
        print('Confusion Matrix:')
        print(confusion_m)
        print(f'Accuracy: {acc} %')

    def _update_best(self, potential_best_individual):
        if potential_best_individual.fitness == np.nan:
            return False
        if self.best_individual is None or self.best_individual.fitness < potential_best_individual.fitness or \
                self.best_individual.fitness == np.nan:
            self.best_individual = potential_best_individual
            logger.info(f'    New best individual ({self.best_individual.key}) found '
                        f'with fitness {round(self.best_individual.fitness, 3)}')
            logger.debug(f'         best individual has {len(self.best_individual.node_genes)} Nodes '
                         f'and {len(self.best_individual.connection_genes)} Connections')
            return True
        return False

    def generate_final_report(self, end_condition):
        self._generate_report(end_condition)
        return self

    def _generate_report(self, end_condition='normal'):
        self.report.add_data(name='generation_metrics', value=self.generation_metrics)
        self.report.add_data(name='best_individual', value=self.get_best_individual().to_dict())
        self.report.add_data(name='best_individual_graph', value=self.get_best_individual().get_graph())
        self.report.add_data(name='best_individual_fitness', value=self.get_best_individual().fitness)
        self.report.add_data(name='fine_tuning', value=self.fine_tuning)
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


def calculate_number_of_parameters(genome):
    n_weight_parameters = 2 * len(genome.connection_genes)
    n_bias_parameters = 2 * len(genome.node_genes)
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
            fitness_all.append(genome.fitness)
            all_n_parameters.append(genome.calculate_number_of_parameters())
            if genome.fitness > self.best_individual_fitness:
                self.best_individual_fitness = genome.fitness
                self.best_individual_key = genome.key

        self.generation_data['best_individual_fitness'] = self.best_individual_fitness
        self.generation_data['best_individual_key'] = self.best_individual_key
        # self.generation_data['best_individual_graph'] = self.population.get(self.best_individual_key).get_graph()
        self.generation_data['all_fitness'] = fitness_all
        self.generation_data['min_fitness'] = round(min(fitness_all), 3)
        self.generation_data['max_fitness'] = round(max(fitness_all), 3)
        self.generation_data['mean_fitness'] = round(np.mean(fitness_all), 3)
        self.generation_data['std_fitness'] = round(np.std(fitness_all), 3)

        # species data
        n_species = len(self.species)
        representative_fitnesses_by_specie = \
            [round(specie.representative.fitness, 3) for specie in self.species.values()]
        best_fitnesses_by_specie = \
            [round(specie.fitness, 3) if specie.fitness else None for specie in self.species.values()]

        n_genomes_by_specie = [len(specie.members) for specie in self.species.values()]
        self.generation_data['n_species'] = n_species
        self.generation_data['representative_fitnesses_by_specie'] = representative_fitnesses_by_specie
        self.generation_data['best_fitnesses_by_specie'] = best_fitnesses_by_specie
        self.generation_data['n_genomes_by_specie'] = n_genomes_by_specie

        # best_n_parameters = self.population.get(self.best_individual_key).calculate_number_of_parameters()
        best_n_parameters = calculate_number_of_parameters(self.population.get(self.best_individual_key))

        logger.info(f'Generation {self.generation}. Best fitness: {round(max(fitness_all), 3)}. '
                    f'N-Parameters Best: {best_n_parameters}')
        logger.info(f'                         Mean fitness: {round(np.mean(fitness_all), 3)}. '
                    f'Mean N-Parameters: {round(np.mean(all_n_parameters), 3)}')
        logger.info(f'                         N-Species: {n_species}.')
        logger.info(f'                                N-Genomes by Specie: {n_genomes_by_specie}')
        logger.info(f'                                Best Fitness by Specie: {best_fitnesses_by_specie}')

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
