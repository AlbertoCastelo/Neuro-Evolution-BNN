from torch import nn
from torch.nn import Module

from neat.configuration import get_configuration, ConfigError


def get_activation(activation=None):
    # if config is None:
    #     config = get_configuration()
    if not activation:
        activation = 'tanh'


    if activation == 'sigmoid':
        return nn.Sigmoid()
    elif activation == 'relu':
        return nn.ReLU()
    elif activation == 'leaky_relu':
        return nn.LeakyReLU()
    elif activation == 'tanh':
        return nn.Tanh()
    elif activation == 'identity':
        return LinearActivation()
    else:
        raise ConfigError(f'Activation function is incorrect: {activation}')


class LinearActivation(nn.Module):

    def forward(self, input):
        return input
