from torch import nn

from neat.configuration import get_configuration, ConfigError


def get_activation():
    config = get_configuration()

    activation = config.node_activation
    if activation == 'sigmoid':
        return nn.Sigmoid()
    elif activation == 'relu':
        return nn.ReLU()
    elif activation == 'leaky_relu':
        return nn.LeakyReLU()
    elif activation == 'tanh':
        return nn.Tanh()
    else:
        raise ConfigError(f'Activation function is incorrect: {activation}')
