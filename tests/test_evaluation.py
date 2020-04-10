import os
from unittest import TestCase

from torch.utils.data import DataLoader

from config_files.configuration_utils import create_configuration
from neat.evaluation.evaluate_simple import evaluate_genome, calculate_prediction_distribution
from neat.evaluation.evaluation_engine import EvaluationStochasticEngine
from neat.evaluation.utils import get_dataset
from neat.genome import Genome
from neat.loss.vi_loss import get_loss
from neat.neat_logger import get_neat_logger
from tests.utils.generate_genome import generate_genome_with_hidden_units
LOGS_PATH = f'{os.getcwd()}/'
logger = get_neat_logger(path=LOGS_PATH)


class TestEvaluationAlternative(TestCase):
    pass
    # def test_happy_path_siso(self):
    #     # Single-Input Single-Output
    #     self.config = create_configuration(filename='/regression-siso.json')
    #     genome = generate_genome_with_hidden_units(n_input=self.config.n_input,
    #                                                n_output=self.config.n_output)
    #
    #     evaluation_engine = EvaluationEngine()
    #
    #     loss = evaluation_engine.evaluate_genome_alternative(genome=genome, n_samples=2)
    #
    #     self.assertEqual(type(loss), float)
    #
    # def test_happy_path_miso(self):
    #     # Multiple-Input Single-Output
    #     self.config = create_configuration(filename='/regression-miso.json')
    #     genome = generate_genome_with_hidden_units(n_input=self.config.n_input,
    #                                                n_output=self.config.n_output)
    #
    #     evaluation_engine = EvaluationAlternativeEngine()
    #
    #     loss = evaluation_engine.evaluate_genome_alternative(genome=genome, n_samples=2)
    #
    #     self.assertEqual(type(loss), float)


class TestEvaluationStochasticNetworkOld(TestCase):

    def test_happy_path_siso(self):
        # Single-Input Single-Output
        self.config = create_configuration(filename='/regression-siso.json')
        genome = generate_genome_with_hidden_units(n_input=self.config.n_input,
                                                   n_output=self.config.n_output)

        evaluation_engine = EvaluationStochasticEngine()

        loss = evaluation_engine.evaluate(genome=genome, n_samples=2)

        self.assertEqual(type(loss), float)

    def test_happy_path_miso(self):
        # Multiple-Input Single-Output
        self.config = create_configuration(filename='/regression-miso.json')
        genome = generate_genome_with_hidden_units(n_input=self.config.n_input,
                                                   n_output=self.config.n_output)

        evaluation_engine = EvaluationStochasticEngine()

        loss = evaluation_engine.evaluate(genome=genome, n_samples=2)

        self.assertEqual(type(loss), float)


class TestEvaluationStochasticNetwork(TestCase):

    def test_happy_path_siso(self):
        # Single-Input Single-Output
        self.config = create_configuration(filename='/regression-siso.json')

        genome = generate_genome_with_hidden_units(n_input=self.config.n_input,
                                                   n_output=self.config.n_output)
        dataset = get_dataset(dataset=self.config.dataset)
        dataset.generate_data()

        data_loader = DataLoader(dataset, batch_size=self.config.batch_size, shuffle=True, num_workers=4)

        loss = get_loss(problem_type=self.config.problem_type)
        evaluate_genome(genome=genome, data_loader=data_loader, loss=loss,
                        beta_type='other',
                        problem_type=self.config.problem_type,
                        batch_size=10000, n_samples=50, is_gpu=False)

        self.assertEqual(type(loss), float)

    def test_happy_path_miso(self):
        # Multiple-Input Single-Output
        self.config = create_configuration(filename='/regression-miso.json')
        genome = generate_genome_with_hidden_units(n_input=self.config.n_input,
                                                   n_output=self.config.n_output)
        dataset = get_dataset(dataset=self.config.dataset)
        dataset.generate_data()

        data_loader = DataLoader(dataset, batch_size=self.config.batch_size, shuffle=True, num_workers=4)

        loss = get_loss(problem_type=self.config.problem_type)
        evaluate_genome(genome=genome, data_loader=data_loader, loss=loss,
                        beta_type='other',
                        problem_type=self.config.problem_type,
                        batch_size=10000, n_samples=50, is_gpu=False)

        self.assertEqual(type(loss), float)


class TestCalculatePredictionDistribution(TestCase):

    def test_regression_case(self):
        config = create_configuration(filename='/regression-siso.json')
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
