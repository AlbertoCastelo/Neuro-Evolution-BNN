from unittest import TestCase

import torch

from deep_learning.probabilistic.deser import ProbabilisticFeedForwardDeser
from deep_learning.probabilistic.feed_forward import ProbabilisticFeedForward


class TestProbabilisticFeedForwardDeser(TestCase):
    def test_round_trip(self):
        n_input = 2
        n_output = 3
        n_neurons_per_layer = 5
        n_hidden_layers = 1
        network = ProbabilisticFeedForward(n_input=n_input,
                                           n_output=n_output,
                                           is_cuda=False,
                                           n_neurons_per_layer=n_neurons_per_layer,
                                           n_hidden_layers=n_hidden_layers)
        network_dict = ProbabilisticFeedForwardDeser.to_dict(network)

        network_reconstructed = ProbabilisticFeedForwardDeser.from_dict(network_dict)

        state_dict = network.state_dict()
        state_dict_reconstructed = network_reconstructed.state_dict()

        for key, key_reconstructed in zip(state_dict, state_dict_reconstructed):
            self.assertEqual(key, key_reconstructed)

            self.assertTrue(torch.allclose(state_dict[key], state_dict_reconstructed[key_reconstructed], atol=1e-02))
