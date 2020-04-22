import os
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, mean_absolute_error
from experiments.reporting.report_repository import ReportRepository
from neat.configuration import set_configuration
from neat.evaluation.evaluate_simple import evaluate_genome
from neat.evaluation.utils import get_dataset
from neat.genome import Genome
from neat.loss.vi_loss import get_loss
from neat.neat_logger import get_neat_logger
import pandas as pd
import numpy as np
import torch

LOGS_PATH = f'{os.getcwd()}/'
logger = get_neat_logger(path=LOGS_PATH)


class ExperimentData:
    def __init__(self, correlation_ids: list, dataset_name, n_samples=1000, project='neuro-evolution',
                 algorithm_version='bayes-neat', keep_top=0.8):
        self.correlation_ids = correlation_ids
        self.dataset_name = dataset_name
        self.n_samples = n_samples
        self.project = project
        self.algorithm_version = algorithm_version
        self.keep_top = keep_top

        self.reports = None
        self.best_genomes = {}
        self.experiment_data = None
        self.configurations = {}

    def process_data(self):
        reports = self.get_reports()
        data_chunks = []
        for report in reports.values():
            execution_id = report.execution_id
            correlation_id = report.correlation_id

            genome_dict = report.data['best_individual']
            best_individual_fitness = report.data['best_individual_fitness']
            print(f'Fitness of best individual: {best_individual_fitness}')

            genome = Genome.from_dict(genome_dict)
            config = genome.genome_config
            self.configurations[execution_id] = config
            self.best_genomes[execution_id] = genome
            set_configuration(config)

            # evaluate genome
            loss = get_loss(problem_type=config.problem_type)
            print(f'Train percentage: {config.train_percentage}')
            print(f'Random state: {config.dataset_random_state}')
            dataset = get_dataset(config.dataset, train_percentage=config.train_percentage, testing=True,
                                  random_state=config.dataset_random_state)

            x, y_true, y_pred, loss_value = evaluate_genome(genome=genome,
                                                            dataset=dataset,
                                                            loss=loss,
                                                            problem_type=config.problem_type,
                                                            beta_type=config.beta_type,
                                                            batch_size=config.batch_size,
                                                            n_samples=self.n_samples,
                                                            is_gpu=config.is_gpu,
                                                            is_testing=True,
                                                            return_all=True)

            y_pred = torch.argmax(y_pred, dim=1)

            train_percentage = config.train_percentage
            noise = config.noise
            duration = report.duration
            n_parameters = genome.calculate_number_of_parameters()
            n_nodes = genome.n_bias_parameters // 2
            n_connections = genome.n_weight_parameters // 2
            mean_genome_std = get_mean_std(genome)
            end_condition = report.data['end_condition']
            chunk = pd.DataFrame({'correlation_id': correlation_id,
                                  'execution_id': execution_id,
                                  'train_percentage': train_percentage,
                                  'noise': noise,
                                  'is_bayesian': False if config.fix_std else True,
                                  'beta': config.beta,
                                  'loss_training': -best_individual_fitness,
                                  'loss_testing': loss_value,
                                  'duration': duration,
                                  'end_condition': end_condition,
                                  'n_parameters': n_parameters,
                                  'n_nodes': n_nodes,
                                  'n_connections': n_connections,
                                  'mean_genome_std': mean_genome_std,
                                  }, index=[0])

            if config.problem_type == 'classification':
                chunk['accuracy'] = accuracy_score(y_true, y_pred) * 100
                chunk['f1'] = f1_score(y_true, y_pred, average='weighted')
            else:
                chunk['mse'] = mean_squared_error(y_true, y_pred)
                chunk['mae'] = mean_absolute_error(y_true, y_pred)

            data_chunks.append(chunk)

        experiment_data = pd.concat(data_chunks, sort=False)

        self.experiment_data = self._drop_worse_executions_per_correlation(experiment_data, self.keep_top)

        return self

    def generate_evolution_data(self):
        fitness_evolution_chunks = []
        fitness_by_specie_chunks = []
        for report in self.get_reports():
            fitness_evolution, fitness_by_specie = self.generate_evolution_data_by_execution(report)
            fitness_evolution_chunks.append(fitness_evolution)
            fitness_by_specie_chunks.append(fitness_by_specie)

        self.fitness_evolution = pd.concat(fitness_evolution_chunks)
        self.fitness_by_specie = pd.concat(fitness_by_specie_chunks)

    def generate_evolution_data_by_execution(self, report):
        generation_metrics = report.data['generation_metrics']
        n_generations = len(generation_metrics)
        fitness_evolution = []
        fitness_by_specie = []
        n_species = 5
        for gen in range(n_generations):
            best_fitness = generation_metrics[str(gen)]['best_individual_fitness']
            worst_fitness = generation_metrics[str(gen)]['min_fitness']
            mean_fitness = generation_metrics[str(gen)]['mean_fitness']
            fitness_evolution.append([gen, best_fitness, worst_fitness, mean_fitness])

            fitness_by_specie_gen = generation_metrics[str(gen)]['genomes_fitness_per_specie']
            mean_fitness_specie_list = [gen]
            for specie, fitnesses_per_specie in fitness_by_specie_gen.items():
                mean_fitness_specie_list.append(np.max(fitnesses_per_specie))

            fitness_by_specie.append(mean_fitness_specie_list)
        fitness_evolution = pd.DataFrame(fitness_evolution,
                                         columns=['generation', 'best_fitness', 'worst_fitness', 'mean_fitness'])
        fitness_evolution['execution_id'] = report.execution_id
        fitness_by_specie = pd.DataFrame(fitness_by_specie,
                                         columns=['generation'] + [f'specie-{i + 1}' for i in range(n_species)])
        fitness_by_specie['execution_id'] = report.execution_id
        return fitness_evolution, fitness_by_specie

    def get_experiment_data(self):
        return self.experiment_data

    def _get_execution_configuration(self, report):
        genome_dict = report.data['best_individual']
        best_individual_fitness = report.data['best_individual_fitness']
        print(f'Fitness of best individual: {best_individual_fitness}')

        genome = Genome.from_dict(genome_dict)
        config = genome.genome_config
        return config

    def get_reports(self):
        if self.reports is None:
            self.reports = self._get_reports()
        return self.reports

    def _get_reports(self):
        reports = {}
        report_repository = ReportRepository.create(project=self.project, logs_path=LOGS_PATH)
        for correlation_id in self.correlation_ids:
            print('###########')
            print(f'CORRELATION ID: {correlation_id}')
            execution_ids = list(report_repository.get_executions(algorithm_version=self.algorithm_version,
                                                                  dataset=self.dataset_name,
                                                                  correlation_id=correlation_id))
            for execution_id in execution_ids:
                report = report_repository.get_report(algorithm_version=self.algorithm_version,
                                                      dataset=self.dataset_name,
                                                      correlation_id=correlation_id,
                                                      execution_id=execution_id)
                reports[execution_id] = report

        return reports

    @staticmethod
    def _drop_worse_executions_per_correlation(experiment_data, keep_top,
                                               filtering_group=('correlation_id', 'noise', 'train_percentage')):
        print(f'Original Size: {len(experiment_data)}')
        experiment_data.sort_values('loss_training', ascending=True, inplace=True)
        executions_per_experiment = experiment_data.groupby(filtering_group)['execution_id'].nunique().reset_index()\
            .rename(columns={'execution_id': 'n_executions'})
        executions_per_experiment['n_executions'] = \
            round(executions_per_experiment['n_executions'] * keep_top, 0)

        experiment_data = experiment_data.merge(executions_per_experiment, on=filtering_group)
        chunks = []
        for filtering_group_values, experiment_data_per_correlation_id in experiment_data.groupby(filtering_group):
            # n_executions = int(executions_per_experiment.loc[
            #                        executions_per_experiment['correlation_id'] == correlation_id,
            #                        'n_executions'].values[0])
            n_executions = int(experiment_data_per_correlation_id['n_executions'].values[0])
            print(n_executions)
            experiment_data_per_correlation_id.sort_values('loss_training', ascending=True, inplace=True)

            chunks.append(experiment_data_per_correlation_id.head(n_executions))

        experiment_data_filtered = pd.concat(chunks, sort=False, ignore_index=True)
        experiment_data_filtered.drop(columns='n_executions', inplace=True)
        print(f'Size after filtering: {len(experiment_data_filtered)}')
        return experiment_data_filtered

def get_mean_std(genome: Genome):
    stds = _get_stds(genome)
    return np.mean(stds)


def _get_stds(genome: Genome):
    stds = []
    for connection in genome.connection_genes.values():
        stds.append(connection._weight_std)
    for bias in genome.node_genes.values():
        stds.append(bias._bias_std)
    return stds
