from unittest import TestCase

from config_files.configuration_utils import create_configuration
from neat.configuration import set_configuration
from neat.evaluation.evaluate_simple import evaluate_genome
from neat.evaluation.utils import get_dataset
from neat.evolution_operators.backprop_mutation import BackPropMutation
from neat.genome import Genome
from neat.loss.vi_loss import get_loss
from neat.representation_mapping.genome_to_network.complex_stochastic_network import DEFAULT_LOGVAR
from tests.representation_mapping.genome_to_network.test_complex_stochastic_network import generate_genome_given_graph


class TestBackpropMutation(TestCase):
    def setUp(self) -> None:
        self.config = create_configuration(filename='/classification-miso.json')
        self.config.node_activation = 'identity'

    def test_normal_feedforward(self):
        genome = generate_genome_given_graph(graph=((-1, 2), (-2, 2), (2, 0), (2, 1)),
                                             connection_weights=(1.0, 2.0, 3.0, 0.5))
        dataset = get_dataset(dataset=self.config.dataset, train_percentage=0.5, testing=False, noise=0.0)
        backprop_mutation = BackPropMutation(dataset=dataset,
                                             n_samples=10,
                                             problem_type=self.config.problem_type,
                                             beta=0.0,
                                             n_epochs=1)
        backprop_mutation.mutate(genome)

        # hidden to output weight matrix is 2 by 1 (with one hidden neuron)
        self.assertEqual(backprop_mutation.network.layer_0.qw_mean.shape, (2, 1))

    def test_network_with_jump(self):
        genome = generate_genome_given_graph(graph=((-1, 2), (-2, 2), (2, 0), (2, 1), (-1, 1)),
                                             connection_weights=(1.0, 2.0, 3.0, 0.5, 1.5))
        dataset = get_dataset(dataset=self.config.dataset, train_percentage=0.5, testing=False, noise=0.0)
        backprop_mutation = BackPropMutation(dataset=dataset,
                                             n_samples=10,
                                             problem_type=self.config.problem_type,
                                             beta=0.0,
                                             n_epochs=1)
        backprop_mutation.mutate(genome)

        # hidden to output weight matrix is 2 by 2 (despite only one hidden neuron)
        self.assertEqual(backprop_mutation.network.layer_0.qw_mean.shape, (2, 2))


class TestClearGradients(TestCase):
    def setUp(self) -> None:
        self.config = create_configuration(filename='/classification-miso.json')
        self.config.node_activation = 'identity'

    def test_non_existing_connections_are_updated(self):
        genome = generate_genome_given_graph(graph=((-1, 2), (-2, 2), (2, 0), (2, 1), (-1, 1)),
                                             connection_weights=(1.0, 2.0, 3.0, 0.5, 1.5))
        dataset = get_dataset(dataset=self.config.dataset, train_percentage=0.1, testing=False, noise=0.0)
        backprop_mutation = BackPropMutation(dataset=dataset,
                                             n_samples=10,
                                             problem_type=self.config.problem_type,
                                             beta=0.0,
                                             n_epochs=2)
        backprop_mutation.mutate(genome)

    def test_network_structure_1(self):
        genome = generate_genome_given_graph(graph=((-1, 2), (-2, 2), (-1, 3),
                                                    (3, 1), (2, 0), (2, 1)),
                                             connection_weights=(1.0, 2.0, 3.0,
                                                                 0.5, -0.5, 0.3))
        dataset = get_dataset(dataset=self.config.dataset, train_percentage=0.1, testing=False, noise=0.0)
        backprop_mutation = BackPropMutation(dataset=dataset,
                                             n_samples=10,
                                             problem_type=self.config.problem_type,
                                             beta=0.0,
                                             n_epochs=2)
        backprop_mutation.mutate(genome)


class TestIntegrationMutation(TestCase):
    def setUp(self) -> None:
        self.config = create_configuration(filename='/classification-miso.json')
        self.config.node_activation = 'identity'

    def test_loss_is_equal_when_no_jumps(self):
        connections = ((-1, 2), (-2, 2), (2, 0), (2, 1))
        genome = generate_genome_given_graph(graph=connections,
                                             connection_weights=(1.0, 2.0, 3.0, 0.5))
        dataset = get_dataset(dataset=self.config.dataset, train_percentage=0.1, testing=False, noise=0.0,
                              random_state=self.config.dataset_random_state)
        n_samples = 100
        backprop_mutation = BackPropMutation(dataset=dataset,
                                             n_samples=n_samples,
                                             problem_type=self.config.problem_type,
                                             beta=0.0,
                                             n_epochs=150)
        genome_mutated = backprop_mutation.mutated_genome(genome)

        # evaluate genome
        loss = get_loss(problem_type=self.config.problem_type)

        loss_value = evaluate_genome(genome=genome_mutated,
                                     dataset=dataset,
                                     loss=loss,
                                     problem_type=self.config.problem_type,
                                     beta_type=self.config.beta_type,
                                     batch_size=self.config.batch_size,
                                     n_samples=n_samples,
                                     is_gpu=False,
                                     is_testing=False,
                                     return_all=False,
                                     is_pass=False)
        self.assertAlmostEqual(loss_value, backprop_mutation.final_loss, places=2)

    def test_non_existing_connections_are_updated(self):
        connections = ((-1, 2), (-2, 2), (2, 0), (2, 1), (-1, 1))
        genome = generate_genome_given_graph(graph=connections,
                                             connection_weights=(1.0, 2.0, 3.0, 0.5, 1.5))
        dataset = get_dataset(dataset=self.config.dataset, train_percentage=0.1, testing=False, noise=0.0)
        backprop_mutation = BackPropMutation(dataset=dataset,
                                             n_samples=10,
                                             problem_type=self.config.problem_type,
                                             beta=0.0,
                                             n_epochs=2)
        genome_mutated = backprop_mutation.mutated_genome(genome)
        self.assertEqual(type(genome_mutated), Genome)
