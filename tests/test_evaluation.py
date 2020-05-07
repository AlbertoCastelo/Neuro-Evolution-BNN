import os
from unittest import TestCase

from config_files.configuration_utils import create_configuration
from neat.evaluation.evaluate_simple import calculate_prediction_distribution
from neat.evaluation.evaluation_engine import EvaluationStochasticEngine
from neat.evaluation.utils import get_dataset
from neat.genome import Genome
from neat.neat_logger import get_neat_logger
from tests.utils.generate_genome import generate_genome_with_hidden_units
LOGS_PATH = f'{os.getcwd()}/'
logger = get_neat_logger(path=LOGS_PATH)


class TestEvaluationStochasticNetwork(TestCase):

    def test_happy_path_miso(self):
        # Multiple-Input Single-Output
        self.config = create_configuration(filename='/classification-miso.json')
        self.config.parallel_evaluation = False
        self.config.n_processes = 1
        genome = generate_genome_with_hidden_units(n_input=self.config.n_input,
                                                   n_output=self.config.n_output)
        population = {1: genome}
        evaluation_engine = EvaluationStochasticEngine()

        population = evaluation_engine.evaluate(population=population)

        self.assertEqual(type(population.get(1).fitness), float)


class TestCalculatePredictionDistribution(TestCase):

    def test_regression_case(self):
        config = create_configuration(filename='/regression-siso.json')
        config.parallel_evaluation = False

        genome = Genome(key=1)
        genome.create_random_genome()

        dataset = get_dataset(config.dataset, train_percentage=config.train_percentage, testing=True)

        n_samples = 3
        x, y_true, output_distribution = calculate_prediction_distribution(genome=genome, dataset=dataset,
                                                                           problem_type=config.problem_type,
                                                                           is_testing=True, n_samples=n_samples,
                                                                           use_sigmoid=False)
        expected_output_distribution_shape = [len(y_true), n_samples, config.n_output]
        self.assertEqual(list(output_distribution.shape), expected_output_distribution_shape)
