from unittest import TestCase

from torch.utils.data import DataLoader

from neat.evaluation.evaluate_simple import evaluate_genome
from neat.evaluation.evaluation_engine import EvaluationStochasticEngine
from neat.evaluation.utils import get_dataset
from neat.loss.vi_loss import get_loss
from tests.config_files.config_files import create_configuration
from tests.utils.generate_genome import generate_genome_with_hidden_units


class TestEvaluationAlternative(TestCase):

    def test_happy_path_siso(self):
        # Single-Input Single-Output
        self.config = create_configuration(filename='/regression-siso.json')
        genome = generate_genome_with_hidden_units(n_input=self.config.n_input,
                                                   n_output=self.config.n_output)

        evaluation_engine = EvaluationAlternativeEngine()

        loss = evaluation_engine.evaluate_genome_alternative(genome=genome, n_samples=2)

        self.assertEqual(type(loss), float)

    def test_happy_path_miso(self):
        # Multiple-Input Single-Output
        self.config = create_configuration(filename='/regression-miso.json')
        genome = generate_genome_with_hidden_units(n_input=self.config.n_input,
                                                   n_output=self.config.n_output)

        evaluation_engine = EvaluationAlternativeEngine()

        loss = evaluation_engine.evaluate_genome_alternative(genome=genome, n_samples=2)

        self.assertEqual(type(loss), float)


class TestEvaluationStochasticNetworkOld(TestCase):

    def test_happy_path_siso(self):
        # Single-Input Single-Output
        self.config = create_configuration(filename='/regression-siso.json')
        genome = generate_genome_with_hidden_units(n_input=self.config.n_input,
                                                   n_output=self.config.n_output)

        evaluation_engine = EvaluationEngine()

        loss = evaluation_engine.evaluate_genome(genome=genome, n_samples=2)

        self.assertEqual(type(loss), float)

    def test_happy_path_miso(self):
        # Multiple-Input Single-Output
        self.config = create_configuration(filename='/regression-miso.json')
        genome = generate_genome_with_hidden_units(n_input=self.config.n_input,
                                                   n_output=self.config.n_output)

        evaluation_engine = EvaluationEngine()

        loss = evaluation_engine.evaluate_genome(genome=genome, n_samples=2)

        self.assertEqual(type(loss), float)


class TestEvaluationStochasticNetwork(TestCase):

    def test_happy_path_siso(self):
        # Single-Input Single-Output
        self.config = create_configuration(filename='/regression-siso.json')

        genome = generate_genome_with_hidden_units(n_input=self.config.n_input,
                                                   n_output=self.config.n_output)
        dataset = get_dataset(dataset_name=self.config.dataset_name)
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
        dataset = get_dataset(dataset_name=self.config.dataset_name)
        dataset.generate_data()

        data_loader = DataLoader(dataset, batch_size=self.config.batch_size, shuffle=True, num_workers=4)

        loss = get_loss(problem_type=self.config.problem_type)
        evaluate_genome(genome=genome, data_loader=data_loader, loss=loss,
                        beta_type='other',
                        problem_type=self.config.problem_type,
                        batch_size=10000, n_samples=50, is_gpu=False)

        self.assertEqual(type(loss), float)
