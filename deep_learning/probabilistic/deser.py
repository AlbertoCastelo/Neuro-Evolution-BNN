from deep_learning.probabilistic.alternative_network.feed_forward_alternative import ProbabilisticFeedForwardAlternative
from deep_learning.probabilistic.feed_forward import ProbabilisticFeedForward
from collections import OrderedDict
import numpy as np
import torch


class ProbabilisticFeedForwardDeser:
    @staticmethod
    def from_dict(network_dict: dict, is_cuda=False, is_alternative=False) -> ProbabilisticFeedForward:
        if is_alternative:
            Network = ProbabilisticFeedForwardAlternative
        else:
            Network = ProbabilisticFeedForward

        network = Network(n_input=network_dict['n_input'],
                          n_output=network_dict['n_output'],
                          is_cuda=is_cuda,
                          n_neurons_per_layer=network_dict['n_neurons_per_layer'],
                          n_hidden_layers=network_dict['n_hidden_layers'])

        # set weights and biases
        network_state = OrderedDict()
        for key, list_ in network_dict['state'].items():
            network_state[key] = torch.Tensor(np.array(list_))
        network.load_state_dict(network_state)
        return network

    @staticmethod
    def to_dict(network: ProbabilisticFeedForward) -> dict:
        network_dict = {'n_input': network.n_input,
                        'n_output': network.n_output,
                        'n_neurons_per_layer': network.n_neurons_per_layer,
                        'n_hidden_layers': network.n_hidden_layers}
        state = {}
        for key, tensor in network.state_dict().items():
            state[key] = tensor.numpy().tolist()
        network_dict['state'] = state
        return network_dict
