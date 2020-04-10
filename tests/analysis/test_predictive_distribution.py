import os
from unittest import TestCase

from config_files.configuration_utils import create_configuration
from neat.analysis.uncertainty.predictive_distribution import PredictionDistributionEstimator
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

        estimator = PredictionDistributionEstimator(genome=genome, config=config, testing=True, n_samples=n_samples)\
            .estimate()

        results = estimator.results
        self.assertTrue(isinstance(results, pd.DataFrame))
