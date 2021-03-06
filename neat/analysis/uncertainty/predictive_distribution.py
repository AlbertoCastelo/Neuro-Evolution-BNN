import torch
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, mean_absolute_error
from torch import nn

from deep_learning.probabilistic.feed_forward import ProbabilisticFeedForward
from neat.configuration import set_configuration
from neat.evaluation.evaluate_simple import calculate_prediction_distribution
from neat.evaluation.utils import get_dataset
from neat.genome import Genome
import pandas as pd
import numpy as np
from experiments.logger import logger
from neat.representation_mapping.genome_to_network.complex_stochastic_network import ComplexStochasticNetwork

EXTREME_QUANTILES = 10
CLASSFICATION_METRICS_DICT = {'accuracy': accuracy_score,
                              'f1': f1_score}
REGRESSION_METRICS_DICT = {'mse': mean_squared_error,
                           'mae': mean_absolute_error}


class PredictionDistributionEstimator:
    def __init__(self, config, testing=True, n_samples=1000):
        self.testing = testing
        self.config = config
        self.n_samples = n_samples

        self.dataset = None

        self.x = None
        self.y_true = None
        self.output_distribution = None
        self.y_pred = None
        self.output_stds = None

        self.results = None
        self.results_enriched = None
        self.metrics_by_quantile = None

    def estimate(self):
        set_configuration(self.config)
        network = self._get_network()
        self._estimate(network)
        return self

    def _get_network(self):
        raise NotImplementedError

    def get_dataset(self):
        if self.dataset is None:
            self.dataset = get_dataset(self.config.dataset, train_percentage=self.config.train_percentage,
                                       testing=self.testing, random_state=self.config.dataset_random_state,
                                       noise=self.config.noise,
                                       label_noise=self.config.label_noise)
        return self.dataset

    def _estimate(self, network: nn.Module):
        x, y_true, output_distribution = calculate_prediction_distribution(network=network,
                                                                           dataset=self.get_dataset(),
                                                                           problem_type=self.config.problem_type,
                                                                           is_testing=self.testing,
                                                                           n_samples=self.n_samples)
        if self.config.problem_type == 'classification':
            y_pred_prob = output_distribution.mean(1)

            y_pred = torch.argmax(y_pred_prob, 1)
            n_classes = output_distribution.shape[2]
            zeros_ = torch.zeros_like(output_distribution.std(1)[:, 1])
            output_stds = torch.zeros_like(output_distribution.std(1)[:, 1])
            for i in range(n_classes):
                output_stds += torch.where(y_pred == i, output_distribution.std(1)[:, i], zeros_)

            # output_stds = output_distribution.std(1)[:, 0]
            min_ = torch.min(output_distribution, 1).values
            max_ = torch.max(output_distribution, 1).values

            zeros_ = torch.zeros_like(output_distribution.std(1)[:, 1])
            output_range = torch.zeros_like(output_distribution.std(1)[:, 1])
            for i in range(n_classes):
                output_range += torch.where(y_pred == i, (max_ - min_)[:, i], zeros_)
            # output_range = (max_ - min_)[:, 0]
        else:
            y_pred = output_distribution.mean(1).squeeze()
            y_true = y_true.squeeze()
            # TODO: review 0 index below
            output_stds = output_distribution.std(1)[:, 0]
            min_ = torch.min(output_distribution, 1).values
            max_ = torch.max(output_distribution, 1).values
            output_range = (max_ - min_)[:, 0]

        results = pd.DataFrame({'y_pred': y_pred.numpy(),
                                'y_true': y_true.numpy(),
                                'std': output_stds.numpy(),
                                'range': output_range.numpy()}).reset_index().rename(columns={'index': 'example_id'})
        results['correct'] = False
        results.loc[(results['y_true'] == results['y_pred']), 'correct'] = True

        self.results = results

        self.x = x
        self.y_pred = y_pred
        self.y_true = y_true
        self.output_distribution = output_distribution
        self.output_stds = output_stds

    def enrich_with_dispersion_quantile(self):
        std_quantiles = self.results[['example_id', 'std']] \
            .sort_values('std', ascending=False) \
            .reset_index(drop=True) \
            .reset_index() \
            .rename(columns={'index': 'order_std'}) \
            .drop('std', axis=1)
        self.results_enriched = self.results.merge(std_quantiles, on='example_id')
        return self

    def calculate_metrics_by_dispersion_quantile(self, log=False):
        if self.config.problem_type == 'classification':
            metrics = CLASSFICATION_METRICS_DICT
        elif self.config.problem_type == 'regression':
            metrics = REGRESSION_METRICS_DICT

        metrics_by_quantile_list = []
        for i in range(0, len(self.results_enriched) - EXTREME_QUANTILES):
            results_filtered = self.results_enriched.loc[self.results_enriched['order_std'] > i]

            mean_std = np.mean(results_filtered['std'].values)
            if log:
                logger.info(f'Mean dispersion: {mean_std} for {len(results_filtered)} points')

            row = [i]
            for metric_name, metric in metrics.items():
                if metric == f1_score:
                    metric_value = metric(results_filtered['y_true'], results_filtered['y_pred'], average='weighted')
                else:
                    metric_value = metric(results_filtered['y_true'], results_filtered['y_pred'])
                row.append(metric_value)

            metrics_by_quantile_list.append(row)

        columns = ['order_std'] + list(metrics.keys())
        self.metrics_by_quantile = pd.DataFrame(metrics_by_quantile_list, columns=columns)
        return self

    def __len__(self):
        return self.output_distribution.shape[0]


class PredictionDistributionEstimatorGenome(PredictionDistributionEstimator):
    def __init__(self, genome: Genome, config, testing=True, n_samples=1000):
        self.genome = genome
        self.is_bayesian = not config.fix_std

        super().__init__(config, testing, n_samples)

    def _get_network(self):
        return ComplexStochasticNetwork(genome=self.genome)


class PredictionDistributionEstimatorNetwork(PredictionDistributionEstimator):
    def __init__(self, network: ProbabilisticFeedForward, config, testing=True, n_samples=1000):
        self.network = network
        self.is_bayesian = True
        super().__init__(config, testing, n_samples)

    def _get_network(self):
        return self.network
