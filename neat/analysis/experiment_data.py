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
                 algorithm_version='bayes-neat'):
        self.correlation_ids = correlation_ids
        self.dataset_name = dataset_name
        self.n_samples = n_samples
        self.project = project
        self.algorithm_version = algorithm_version

        self.best_genomes = {}
        self.experiment_data = None
        self.configurations = {}

    def process_data(self):
        report_repository = ReportRepository.create(project=self.project, logs_path=LOGS_PATH)
        data_chunks = []
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
                print(config.train_percentage)
                dataset = get_dataset(config.dataset, train_percentage=config.train_percentage, testing=True)

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
                duration = report.duration
                n_parameters = genome.calculate_number_of_parameters()
                n_nodes = genome.n_bias_parameters // 2
                n_connections = genome.n_weight_parameters // 2
                mean_genome_std = get_mean_std(genome)
                end_condition = report.data['end_condition']
                chunk = pd.DataFrame({'correlation_id': correlation_id,
                                      'execution_id': execution_id,
                                      'train_percentage': train_percentage,
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
                    chunk['f1'] = f1_score(y_true, y_pred)
                else:
                    chunk['mse'] = mean_squared_error(y_true, y_pred)
                    chunk['mae'] = mean_absolute_error(y_true, y_pred)

                data_chunks.append(chunk)
            print(config.fix_std)
            print(config.beta)

        self.experiment_data = pd.concat(data_chunks, sort=False)
        return self

    def get_experiment_data(self):
        return self.experiment_data


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
