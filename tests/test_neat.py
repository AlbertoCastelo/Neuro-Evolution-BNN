from unittest import TestCase
import os
from unittest.mock import Mock

from config_files.configuration_utils import create_configuration
from neat.evolution_operators.backprop_mutation import BACKPROP_MUTATION
from neat.evolution_operators.mutation import RANDOM_MUTATION
from neat.neat_logger import get_neat_logger
from neat.population_engine import EvolutionEngine

LOGS_PATH = f'{os.getcwd()}/'
logger = get_neat_logger(path=LOGS_PATH)


class TestIntegrationNeat(TestCase):
    def setUp(self) -> None:
        self.report = Mock()
        self.notifier = Mock()
        self.config = create_configuration(filename='/classification-miso.json')
        self.config.node_activation = 'identity'
        self.config.n_generations = 1
        self.config.pop_size = 20

    def test_neat_random_mutation_runs_happy_path(self):
        self.config.mutation_type = RANDOM_MUTATION
        self.config.parallel_evaluation = False
        self.config.is_fine_tuning = False

        evolution_engine = EvolutionEngine(report=self.report, notifier=self.notifier)
        evolution_engine.run()

    def test_neat_backprop_mutation_runs_happy_path(self):
        self.config.mutation_type = BACKPROP_MUTATION
        self.config.parallel_evaluation = False
        self.config.is_fine_tuning = False

        evolution_engine = EvolutionEngine(report=self.report, notifier=self.notifier)
        evolution_engine.run()

    def test_final_fine_tuning(self):
        self.config.mutation_type = RANDOM_MUTATION
        self.config.is_fine_tuning = True
        self.config.parallel_evaluation = False
        self.config.epochs_fine_tuning = 2
        evolution_engine = EvolutionEngine(report=self.report, notifier=self.notifier)
        evolution_engine.run()
