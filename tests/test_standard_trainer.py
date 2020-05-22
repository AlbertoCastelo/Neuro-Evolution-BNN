from unittest import TestCase

from config_files.configuration_utils import create_configuration
from neat.evaluation.evaluate_simple import evaluate_genome
from neat.evaluation.utils import get_dataset
from neat.genome import Genome
from neat.loss.vi_loss import get_loss
from neat.representation_mapping.genome_to_network.complex_stochastic_network import ComplexStochasticNetwork, equal
from neat.representation_mapping.network_to_genome.stochastic_network_to_genome import \
    convert_stochastic_network_to_genome
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
                                      n_output=genome.n_output,
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
                                      n_output=genome.n_output,
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
                                      n_output=genome.n_output,
                                      problem_type=self.config.problem_type,
                                      beta=self.config.beta,
                                      n_epochs=n_epochs, is_cuda=False)
        stg_trainer.train(genome)
        best_network = stg_trainer.get_best_network()
        self.assertEqual(type(best_network), ComplexStochasticNetwork)
        self.assertNotEqual(str(stg_trainer.best_loss_val), 'nan')
        self.assertNotEqual(str(stg_trainer.final_loss), 'nan')


class TestClearGradients(TestCase):
    def setUp(self) -> None:
        self.config = create_configuration(filename='/classification-miso.json')
        self.config.node_activation = 'identity'
        self.n_epochs = 2

    def test_non_existing_connections_are_updated(self):
        genome = generate_genome_given_graph(graph=((-1, 2), (-2, 2), (2, 0), (2, 1), (-1, 1)),
                                             connection_weights=(1.0, 2.0, 3.0, 0.5, 1.5))
        dataset = get_dataset(dataset=self.config.dataset, train_percentage=0.1, testing=False, noise=0.0)
        stg_trainer = StandardTrainer(dataset=dataset,
                                      n_samples=self.config.n_samples,
                                      n_output=genome.n_output,
                                      problem_type=self.config.problem_type,
                                      beta=self.config.beta,
                                      n_epochs=self.n_epochs,
                                      is_cuda=False)
        stg_trainer.train(genome)

    def test_network_structure_1(self):
        genome = generate_genome_given_graph(graph=((-1, 2), (-2, 2), (-1, 3),
                                                    (3, 1), (2, 0), (2, 1)),
                                             connection_weights=(1.0, 2.0, 3.0,
                                                                 0.5, -0.5, 0.3))
        dataset = get_dataset(dataset=self.config.dataset, train_percentage=0.1, testing=False, noise=0.0)
        stg_trainer = StandardTrainer(dataset=dataset,
                                      n_samples=self.config.n_samples,
                                      problem_type=self.config.problem_type,
                                      beta=self.config.beta,
                                      n_output=genome.n_output,
                                      n_epochs=self.n_epochs,
                                      is_cuda=False)
        stg_trainer.train(genome)


class TestIntegrationMutation(TestCase):
    def setUp(self) -> None:
        self.config = create_configuration(filename='/classification-miso.json')
        self.config.node_activation = 'identity'
        self.config.fix_std = False
        self.n_epochs = 1

    def test_loss_is_equal_when_no_jumps(self):
        connections = ((-1, 2), (-2, 2), (2, 0), (2, 1))
        genome = generate_genome_given_graph(graph=connections,
                                             connection_weights=(1.0, 2.0, 3.0, 0.5))
        dataset = get_dataset(dataset=self.config.dataset, train_percentage=0.1, testing=False, noise=0.0,
                              random_state=self.config.dataset_random_state)
        n_samples = 100
        stg_trainer = StandardTrainer(dataset=dataset, n_samples=self.config.n_samples,
                                      n_output=genome.n_output,
                                      problem_type=self.config.problem_type,
                                      beta=self.config.beta,
                                      n_epochs=100, is_cuda=False)
        stg_trainer.train(genome)
        best_network = stg_trainer.get_best_network()
        new_genome = convert_stochastic_network_to_genome(network=best_network,
                                                          original_genome=genome,
                                                          fitness=-stg_trainer.best_loss_val,
                                                          fix_std=genome.genome_config.fix_std)

        # evaluate genome
        loss = get_loss(problem_type=self.config.problem_type)

        loss_value = evaluate_genome(genome=new_genome,
                                     dataset=dataset,
                                     loss=loss,
                                     problem_type=self.config.problem_type,
                                     beta_type=self.config.beta_type,
                                     batch_size=self.config.batch_size,
                                     n_samples=n_samples,
                                     is_gpu=False,
                                     is_testing=False,
                                     return_all=False,
                                     is_pass=True)
        self.assertAlmostEqual(loss_value, -new_genome.fitness, places=0)

    def test_non_existing_connections_are_updated(self):
        connections = ((-1, 2), (-2, 2), (2, 0), (2, 1), (-1, 1))
        genome = generate_genome_given_graph(graph=connections,
                                             connection_weights=(1.0, 2.0, 3.0, 0.5, 1.5))
        dataset = get_dataset(dataset=self.config.dataset, train_percentage=0.1, testing=False, noise=0.0)

        stg_trainer = StandardTrainer(dataset=dataset, n_samples=self.config.n_samples,
                                      n_output=genome.n_output,
                                      problem_type=self.config.problem_type,
                                      beta=self.config.beta,
                                      n_epochs=self.n_epochs, is_cuda=False)
        stg_trainer.train(genome)
        best_network = stg_trainer.get_best_network()
        new_genome = convert_stochastic_network_to_genome(network=best_network,
                                                          original_genome=genome,
                                                          fitness=-stg_trainer.best_loss_val,
                                                          fix_std=genome.genome_config.fix_std)

        network_mutated = ComplexStochasticNetwork(genome=new_genome)

        self.assertEqual(type(new_genome), Genome)
        self.assertTrue(equal(best_network, network_mutated))

    def test_non_existing_connections_are_updated_2(self):
        connections = ((-1, 1), (-2, 1))
        genome = generate_genome_given_graph(graph=connections,
                                             connection_weights=(1.0, 2.0))
        dataset = get_dataset(dataset=self.config.dataset, train_percentage=0.1, testing=False, noise=0.0)
        stg_trainer = StandardTrainer(dataset=dataset, n_samples=self.config.n_samples,
                                      n_output=genome.n_output,
                                      problem_type=self.config.problem_type,
                                      beta=self.config.beta,
                                      n_epochs=self.n_epochs, is_cuda=False)
        stg_trainer.train(genome)
        best_network = stg_trainer.get_best_network()
        new_genome = convert_stochastic_network_to_genome(network=best_network,
                                                          original_genome=genome,
                                                          fitness=-stg_trainer.best_loss_val,
                                                          fix_std=genome.genome_config.fix_std)
        network_mutated = ComplexStochasticNetwork(genome=new_genome)

        self.assertEqual(type(new_genome), Genome)
        self.assertTrue(equal(best_network, network_mutated))

