import numpy as np

from neat.configuration import get_configuration

NODE_TYPE = 'node'
CONNECTION_TYPE = 'connection'


class Gene:

    def __init__(self, key, type):
        self.key = key
        self.type = type


class ConnectionGene(Gene):
    main_attributes = ['weight_mean', 'weight_std']

    def __init__(self, key):
        '''
        key: must be a tuple of nodes' keys (key-origin-node, key-destiny-node)
        '''
        if not isinstance(key, tuple):
            raise ValueError('Key needs to be a tuple')
        super().__init__(key=key, type=CONNECTION_TYPE)
        self.key = key
        self.enabled = True
        self.config = get_configuration()
        self.weight_configuration = WeightConfig()

        self.weight_mean = None
        self.weight_std = None
        self.weight_log_var = None

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

    def take_sample(self):
        return np.random.normal(loc=self.weight_mean, scale=self.weight_std)


class NodeGene(Gene):
    main_attributes = ['bias_mean', 'bias_std']

    def __init__(self, key):
        super().__init__(key=key, type=NODE_TYPE)
        self.key = key
        self.config = get_configuration()

        self.activation = self.config.node_activation
        self.aggregation = self.config.node_aggregation

        self.bias_configuration = BiasConfig()
        self.bias_mean = None
        self.bias_std = None
        self.bias_log_var = None

    def random_initialization(self):
        mean = self.bias_configuration.bias_mean_init_mean
        std = self.bias_configuration.bias_mean_init_std
        bias = np.random.normal(loc=mean, scale=std)
        bias = np.clip(bias,
                       a_min=self.bias_configuration.bias_mean_min_value,
                       a_max=self.bias_configuration.bias_mean_max_value)
        self.bias_mean = bias

        bias_std_mean = self.bias_configuration.bias_std_init_mean
        bias_std_std = self.bias_configuration.bias_std_init_std

        bias_std = np.random.normal(loc=bias_std_mean, scale=bias_std_std)
        bias_std = np.clip(bias_std,
                           a_min=self.bias_configuration.bias_std_min_value,
                           a_max=self.bias_configuration.bias_std_max_value)
        self.bias_std = bias_std

    def take_sample(self):
        return np.random.normal(loc=self.bias_mean, scale=self.bias_std)


class BiasConfig:

    def __init__(self):
        config = get_configuration()
        self.bias_mean_init_mean = config.bias_mean_init_mean
        self.bias_mean_init_std = config.bias_mean_init_std

        self.bias_std_init_mean = config.bias_std_init_mean
        self.bias_std_init_std = config.bias_std_init_std

        self.bias_mean_max_value = config.bias_mean_max_value
        self.bias_mean_min_value = config.bias_mean_min_value

        self.bias_std_max_value = config.bias_std_max_value
        self.bias_std_min_value = config.bias_std_min_value

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

