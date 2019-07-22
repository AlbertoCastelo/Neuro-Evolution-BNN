from unittest import TestCase

from neat.evaluation import EvaluationEngine
from tests.utils import generate_genome_with_hidden_units


class TestEvaluationAlternative(TestCase):

    def test_happy_path(self):
        genome = generate_genome_with_hidden_units()

        evaluation_engine = EvaluationEngine()

        loss = evaluation_engine.evaluate_genome_alternative(genome=genome, n_samples=2)

        self.assertEqual(type(loss), float)
