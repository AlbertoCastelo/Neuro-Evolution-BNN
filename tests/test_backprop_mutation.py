from unittest import TestCase

from config_files.configuration_utils import create_configuration
from neat.evaluation.utils import get_dataset
from neat.evolution_operators.backprop_mutation import BackPropMutation
from tests.representation_mapping.genome_to_network.test_complex_stochastic_network import generate_genome_given_graph


class TestBackpropMutation(TestCase):
    def setUp(self) -> None:
        self.config = create_configuration(filename='/classification-miso.json')
        self.config.node_activation = 'identity'
        self.config.parallel_evaluation = False

    def test_normal_feedforward(self):
        genome = generate_genome_given_graph(graph=((-1, 2), (-2, 2), (2, 0), (2, 1)),
                                             connection_weights=(1.0, 2.0, 3.0, 0.5))
        dataset = get_dataset(dataset=self.config.dataset, train_percentage=0.5, testing=False, noise=0.0)
        backprop_mutation = BackPropMutation(dataset=dataset,
                                             n_samples=10,
                                             problem_type=self.config.problem_type,
                                             beta=0.0,
                                             n_epochs=1)
        mutated_genome = backprop_mutation.mutate(genome)

        self.assertTrue(mutated_genome.check_same_architecture(genome))

    def test_network_with_jump(self):
        genome = generate_genome_given_graph(graph=((-1, 2), (-2, 2), (2, 0), (2, 1), (-1, 1)),
                                             connection_weights=(1.0, 2.0, 3.0, 0.5, 1.5))
        dataset = get_dataset(dataset=self.config.dataset, train_percentage=0.5, testing=False, noise=0.0)
        backprop_mutation = BackPropMutation(dataset=dataset,
                                             n_samples=10,
                                             problem_type=self.config.problem_type,
                                             beta=0.0,
                                             n_epochs=1)
        mutated_genome = backprop_mutation.mutate(genome)

        self.assertTrue(mutated_genome.check_same_architecture(genome))
