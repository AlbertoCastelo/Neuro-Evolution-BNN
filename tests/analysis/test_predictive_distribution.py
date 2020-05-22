import os
from unittest import TestCase

from config_files.configuration_utils import create_configuration
from deep_learning.probabilistic.feed_forward import ProbabilisticFeedForward
from neat.analysis.uncertainty.predictive_distribution import \
    PredictionDistributionEstimatorGenome, PredictionDistributionEstimatorNetwork
import pandas as pd
from neat.genome import Genome
from neat.neat_logger import get_neat_logger

LOGS_PATH = f'{os.getcwd()}/'
logger = get_neat_logger(path=LOGS_PATH)


class TestPredictionDistributionEstimator(TestCase):

    def test_regression_case(self):
        config = create_configuration(filename='/regression-siso.json')
        genome = Genome(key=1)
        genome.create_random_genome()
        n_samples = 3

        estimator = PredictionDistributionEstimatorGenome(genome=genome, config=config, testing=True,
                                                          n_samples=n_samples)\
            .estimate() \
            .enrich_with_dispersion_quantile() \
            .calculate_metrics_by_dispersion_quantile()

        results = estimator.results
        self.assertTrue(isinstance(results, pd.DataFrame))

    def test_network_case(self):
        config = create_configuration(filename='/regression-siso.json')
        n_samples = 3
        network = ProbabilisticFeedForward(1, 1, False, 1, 1)
        estimator = PredictionDistributionEstimatorNetwork(network=network, config=config, testing=True,
                                                           n_samples=n_samples)\
            .estimate() \
            .enrich_with_dispersion_quantile() \
            .calculate_metrics_by_dispersion_quantile()

        results = estimator.results
        self.assertTrue(isinstance(results, pd.DataFrame))

    def test_classification_case(self):
        config = create_configuration(filename='/classification-miso.json')
        genome = Genome(key=1)
        genome.create_random_genome()
        n_samples = 5

        estimator = PredictionDistributionEstimatorGenome(genome=genome, config=config, testing=True,
                                                          n_samples=n_samples)\
            .estimate() \
            .enrich_with_dispersion_quantile() \
            .calculate_metrics_by_dispersion_quantile()

        results = estimator.results
        self.assertTrue(isinstance(results, pd.DataFrame))
