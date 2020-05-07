from unittest import TestCase

from config_files.configuration_utils import create_configuration
from neat.evaluation.utils import get_dataset
from neat.representation_mapping.genome_to_network.complex_stochastic_network import ComplexStochasticNetwork
from neat.standard_training.standard_trainer import StandardTrainer
from tests.representation_mapping.genome_to_network.test_complex_stochastic_network import generate_genome_given_graph
import numpy as np


class TestStandardTrainer(TestCase):
    def setUp(self) -> None:
        self.config = create_configuration(filename='/classification-miso.json')
        self.config.node_activation = 'identity'
        self.config.parallel_evaluation = False
        self.config.n_samples = 2
        self.config.beta = 0.0

    def test_normal_feedforward(self):
        n_epochs = 2
        genome = generate_genome_given_graph(graph=((-1, 2), (-2, 2), (2, 0), (2, 1)),
                                             connection_weights=(1.0, 2.0, 3.0, 0.5))
        dataset = get_dataset(dataset=self.config.dataset, train_percentage=0.5, testing=False, noise=0.0)
        stg_trainer = StandardTrainer(dataset=dataset, n_samples=self.config.n_samples,
                                      problem_type=self.config.problem_type,
                                      beta=self.config.beta,
                                      n_epochs=n_epochs, is_cuda=False)
        stg_trainer.train(genome)
        best_network = stg_trainer.get_best_network()
        self.assertEqual(type(best_network), ComplexStochasticNetwork)
        self.assertNotEqual(str(stg_trainer.best_loss_val), 'nan')
        self.assertNotEqual(str(stg_trainer.final_loss), 'nan')

    def test_network_with_jump(self):
        n_epochs = 2
        genome = generate_genome_given_graph(graph=((-1, 2), (-2, 2), (2, 0), (2, 1), (-1, 1)),
                                             connection_weights=(1.0, 2.0, 3.0, 0.5, 1.5))
        dataset = get_dataset(dataset=self.config.dataset, train_percentage=0.5, testing=False, noise=0.0)
        stg_trainer = StandardTrainer(dataset=dataset, n_samples=self.config.n_samples,
                                      problem_type=self.config.problem_type,
                                      beta=self.config.beta,
                                      n_epochs=n_epochs, is_cuda=False)
        stg_trainer.train(genome)
        best_network = stg_trainer.get_best_network()
        self.assertEqual(type(best_network), ComplexStochasticNetwork)
        self.assertNotEqual(str(stg_trainer.best_loss_val), 'nan')
        self.assertNotEqual(str(stg_trainer.final_loss), 'nan')

    def test_network_scarce(self):
        n_epochs = 2
        genome = generate_genome_given_graph(graph=((-2, 2), (-1, 1)),
                                             connection_weights=(1.0, 2.0))
        dataset = get_dataset(dataset=self.config.dataset, train_percentage=0.5, testing=False, noise=0.0)
        stg_trainer = StandardTrainer(dataset=dataset, n_samples=self.config.n_samples,
                                      problem_type=self.config.problem_type,
                                      beta=self.config.beta,
                                      n_epochs=n_epochs, is_cuda=False)
        stg_trainer.train(genome)
        best_network = stg_trainer.get_best_network()
        self.assertEqual(type(best_network), ComplexStochasticNetwork)
        self.assertNotEqual(str(stg_trainer.best_loss_val), 'nan')
        self.assertNotEqual(str(stg_trainer.final_loss), 'nan')

