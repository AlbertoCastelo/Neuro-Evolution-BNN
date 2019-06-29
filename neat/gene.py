import numpy as np

from neat.configuration import get_configuration

NODE_TYPE = 'node'
CONNECTION_TYPE = 'connection'


class Gene:

    def __init__(self, key, type):
        self.key = key
        self.type = type


class ConnectionGene(Gene):

    def __init__(self, key):
        super().__init__(key=key, type=CONNECTION_TYPE)
        self.key = key
        self.enabled = True
        self.config = get_configuration()
        self.weight_configuration = WeightConfig()

        self.weight_mean = None
        self.weight_std = None

    def random_initialization(self):
        weight_mean_mean = self.weight_configuration.weight_mean_init_mean
        weight_mean_std = self.weight_configuration.weight_mean_init_std

        weight_mean = np.random.normal(loc=weight_mean_mean, scale=weight_mean_std)
        weight_mean = np.clip(weight_mean,
                              a_min=self.weight_configuration.weight_mean_min_value,
                              a_max=self.weight_configuration.weight_mean_max_value)
        self.weight_mean = weight_mean

        weight_std_mean = self.weight_configuration.weight_std_init_mean
        weight_std_std = self.weight_configuration.weight_std_init_std

        weight_std = np.random.normal(loc=weight_std_mean, scale=weight_std_std)
        weight_std = np.clip(weight_std,
                             a_min=self.weight_configuration.weight_std_min_value,
                             a_max=self.weight_configuration.weight_std_max_value)
        self.weight_std = weight_std



class NodeGene(Gene):

    def __init__(self, key):
        super().__init__(key=key, type=NODE_TYPE)
        self.key = key
        self.config = get_configuration()

        self.activation = self.config.node_activation
        self.aggregation = self.config.node_aggregation

        self.bias_configuration = BiasConfig()
        self.bias = None

    def random_initialization(self):
        mean = self.bias_configuration.bias_init_mean
        std = self.bias_configuration.bias_init_std
        bias = np.random.normal(loc=mean, scale=std)
        bias = np.clip(bias,
                       a_min=self.bias_configuration.bias_min_value,
                       a_max=self.bias_configuration.bias_max_value)
        self.bias = bias

class BiasConfig:

    def __init__(self):
        config = get_configuration()
        self.bias_init_mean = config.bias_init_mean
        self.bias_init_std = config.bias_init_std
        self.bias_max_value = config.bias_max_value
        self.bias_min_value = config.bias_min_value
        self.bias_mutate_power = config.bias_mutate_power
        self.bias_mutate_rate = config.bias_mutate_rate
        self.bias_replace_rate = config.bias_replace_rate


class WeightConfig:
    def __init__(self):
        config = get_configuration()
        self.weight_mean_init_mean = config.weight_mean_init_mean
        self.weight_mean_init_std = config.weight_mean_init_std
        self.weight_mean_max_value = config.weight_mean_max_value
        self.weight_mean_min_value = config.weight_mean_min_value

        self.weight_std_init_mean = config.weight_std_init_mean
        self.weight_std_init_std = config.weight_std_init_std
        self.weight_std_max_value = config.weight_std_max_value
        self.weight_std_min_value = config.weight_std_min_value

        weight_mutate_power = 0.5
        weight_mutate_rate = 0.8
        weight_replace_rate = 0.1

