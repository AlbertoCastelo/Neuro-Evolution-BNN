from torch import nn

from neat.representation_mapping.genome_to_network.layers import StochasticLinear
from neat.representation_mapping.genome_to_network.utils import get_activation


class ProbabilisticFeedForward(nn.Module):
    def __init__(self, n_input, n_output, n_neurons_per_layer=3, n_hidden_layers=2):
        super(ProbabilisticFeedForward, self).__init__()
        self.n_neurons_per_layer = n_neurons_per_layer
        self.n_hidden_layers = n_hidden_layers
        self.activation = get_activation()
        self.n_layers = n_hidden_layers
        in_features = n_input

        # hidden layers
        for i in range(n_hidden_layers, 0, -1):
            # layer = nn.Linear(in_features=in_features, out_features=n_neurons_per_layer)
            layer = StochasticLinear(in_features=in_features, out_features=n_neurons_per_layer)
            setattr(self, f'layer_{i}', layer)
            setattr(self, f'activation_{i}', self.activation)
            in_features = n_neurons_per_layer

        # output layer
        # print(in_features)
        # print(n_output)
        layer = StochasticLinear(in_features=in_features, out_features=n_output)
        setattr(self, f'layer_0', layer)

    def forward(self, x):
        kl_qw_pw = 0.0
        start_index = self.n_layers
        for i in range(start_index, -1, -1):
            # print(x)
            # print(f'Calculating layer {i}')
            x, kl_layer = getattr(self, f'layer_{i}')(x)
            # print(x)
            # print(kl_layer)
            kl_qw_pw += kl_layer
            if i > 0:
                x = getattr(self, f'activation_{i}')(x)
        return x, kl_qw_pw
