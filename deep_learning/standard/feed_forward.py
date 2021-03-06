import torch
from torch import nn
from neat.representation_mapping.genome_to_network.utils import get_activation


class FeedForward(nn.Module):
    def __init__(self, n_input, n_output, n_neurons_per_layer=3, n_hidden_layers=2):
        super(FeedForward, self).__init__()
        self.n_input = n_input
        self.n_output = n_output
        self.n_neurons_per_layer = n_neurons_per_layer
        self.n_hidden_layers = n_hidden_layers
        self.activation = get_activation()
        self.n_layers = n_hidden_layers
        in_features = n_input

        # hidden layers
        for i in range(n_hidden_layers, 0, -1):
            layer = nn.Linear(in_features=in_features, out_features=n_neurons_per_layer)
            setattr(self, f'layer_{i}', layer)
            setattr(self, f'activation_{i}', self.activation)
            in_features = n_neurons_per_layer

        # output layer
        layer = nn.Linear(in_features=in_features, out_features=n_output)
        setattr(self, f'layer_0', layer)

    def forward(self, x):
        for i in range(self.n_layers, 0, -1):
            x = getattr(self, f'layer_{i}')(x)
            x = getattr(self, f'activation_{i}')(x)
        x = getattr(self, f'layer_0')(x)
        return x

    def reset_parameters(self):
        for i in range(self.n_hidden_layers, -1, -1):
            layer = getattr(self, f'layer_{i}')
            torch.nn.init.xavier_uniform(layer.weight)
            layer.bias.data.fill_(0.01)
