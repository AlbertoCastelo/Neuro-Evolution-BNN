from unittest import TestCase, skip

from neat.evaluation import EvaluationEngine, EvaluationAlternativeEngine, EvaluationStochasticGoodEngine
from tests.config_files.config_files import create_configuration
from tests.utils.generate_genome import generate_genome_with_hidden_units


class TestEvaluationAlternative(TestCase):

    def test_happy_path_siso(self):
        # Single-Input Single-Output
        self.config = create_configuration(filename='/siso.json')
        genome = generate_genome_with_hidden_units(n_input=self.config.n_input,
                                                   n_output=self.config.n_output)

        evaluation_engine = EvaluationAlternativeEngine()

        loss = evaluation_engine.evaluate_genome_alternative(genome=genome, n_samples=2)

        self.assertEqual(type(loss), float)

    def test_happy_path_miso(self):
        # Multiple-Input Single-Output
        self.config = create_configuration(filename='/miso.json')
        genome = generate_genome_with_hidden_units(n_input=self.config.n_input,
                                                   n_output=self.config.n_output)

        evaluation_engine = EvaluationAlternativeEngine()

        loss = evaluation_engine.evaluate_genome_alternative(genome=genome, n_samples=2)

        self.assertEqual(type(loss), float)


class TestEvaluationStochasticNetworkOld(TestCase):

    def test_happy_path_siso(self):
        # Single-Input Single-Output
        self.config = create_configuration(filename='/siso.json')
        genome = generate_genome_with_hidden_units(n_input=self.config.n_input,
                                                   n_output=self.config.n_output)

        evaluation_engine = EvaluationEngine()

        loss = evaluation_engine.evaluate_genome(genome=genome, n_samples=2)

        self.assertEqual(type(loss), float)

    def test_happy_path_miso(self):
        # Multiple-Input Single-Output
        self.config = create_configuration(filename='/miso.json')
        genome = generate_genome_with_hidden_units(n_input=self.config.n_input,
                                                   n_output=self.config.n_output)

        evaluation_engine = EvaluationEngine()

        loss = evaluation_engine.evaluate_genome(genome=genome, n_samples=2)

        self.assertEqual(type(loss), float)


class TestEvaluationStochasticNetwork(TestCase):

    def test_happy_path_siso(self):
        # Single-Input Single-Output
        self.config = create_configuration(filename='/siso.json')
        genome = generate_genome_with_hidden_units(n_input=self.config.n_input,
                                                   n_output=self.config.n_output)

        evaluation_engine = EvaluationStochasticGoodEngine()

        loss = evaluation_engine.evaluate_genome(genome=genome, n_samples=2)

        self.assertEqual(type(loss), float)

    def test_happy_path_miso(self):
        # Multiple-Input Single-Output
        self.config = create_configuration(filename='/miso.json')
        genome = generate_genome_with_hidden_units(n_input=self.config.n_input,
                                                   n_output=self.config.n_output)

        evaluation_engine = EvaluationStochasticGoodEngine()

        loss = evaluation_engine.evaluate_genome(genome=genome, n_samples=2)

        self.assertEqual(type(loss), float)