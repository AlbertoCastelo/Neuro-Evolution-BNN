import copy
import random

import numpy as np

from neat.configuration import get_configuration

NODE_TYPE = 'node'
CONNECTION_TYPE = 'connection'
PARAMETERS_NAMES = {NODE_TYPE: 'bias',
                    CONNECTION_TYPE: 'weight'}

CROSSOVER_ATTRIBUTES = ['mean', 'std']
MUTATION_ATTRIBUTES = ['mean']


class Gene:

    def __init__(self, key, type):
        self.key = key
        self.type = type
        self.parameter_name = PARAMETERS_NAMES[type]
        self.crossover_attributes = copy.deepcopy(CROSSOVER_ATTRIBUTES)
        self.mutation_attributes = copy.deepcopy(MUTATION_ATTRIBUTES)

        self.config = get_configuration()
        self.is_discrete = self.config.is_discrete
        self.fix_std = self.config.fix_std
        if not self.fix_std:
            self.mutation_attributes.append('std')

        self.single_structural_mutation = self.config.single_structural_mutation
        self.mutate_rate = self.config.mutate_rate
        self.mutate_power = self.config.mutate_power
        self.replace_rate = self.config.replace_rate

        self.mean_name = f'_{self.parameter_name}_mean'
        self.var_name = f'_{self.parameter_name}_var'
        self.std_name = f'_{self.parameter_name}_std'
        self.log_var_name = f'_{self.parameter_name}_log_var'

        setattr(self, self.mean_name, None)
        setattr(self, self.var_name, None)
        setattr(self, self.log_var_name, None)
        setattr(self, self.std_name, None)

    def mutate(self):
        for attribute in self.mutation_attributes:
            attribute_value = getattr(self, f'get_{attribute}')()
            mutated_value = self._mutate_float(value=attribute_value, parameter_name=self.parameter_name,
                                               parameter_type=attribute)
            getattr(self, f'set_{attribute}')(mutated_value)

    def set_std(self, std):
        std = self._clip(value=std, parameter_type='std')
        self._set_std(std)
        var = calculate_variance_given_std(std)
        self._set_variance(var)
        log_var = calculate_log_var_given_variance(var)
        self._set_log_var(log_var)

    def set_log_var(self, log_var):
        self._set_log_var(log_var)
        var = calculate_variance_given_log_var(log_var)
        self._set_variance(var)
        std = calculate_std_given_variance(var)
        std = self._clip(value=std, parameter_type='std')
        self._set_std(std)

    def set_mean(self, mean):
        mean = self._clip(value=mean, parameter_type='mean')
        setattr(self, self.mean_name, mean)

    def get_mean(self):
        return getattr(self, self.mean_name)

    def get_std(self):
        return getattr(self, self.std_name)

    def get_variance(self):
        return getattr(self, self.var_name)

    def get_log_var(self):
        return getattr(self, self.log_var_name)

    def _set_log_var(self, log_var):
        setattr(self, self.log_var_name, log_var)

    def _set_variance(self, var):
        assert var >= 0.0
        setattr(self, self.var_name, var)

    def _set_std(self, std):
        assert std >= 0.0
        setattr(self, self.std_name, std)

    def _set_mean(self, mean):
        setattr(self, self.mean_name, mean)

    def _clip(self, value: float, parameter_type: str) -> float:
        min_ = getattr(self.config, f'{self.parameter_name}_{parameter_type}_min_value')
        max_ = getattr(self.config, f'{self.parameter_name}_{parameter_type}_max_value')
        return float(np.clip(value,
                             a_min=min_,
                             a_max=max_))

    def _initialize_float(self, parameter_name: str, parameter_type: str):
        if self.is_discrete:
            return self._sample_discrete_float(parameter_name, parameter_type)
        return self._initialize_continuous_float(parameter_name, parameter_type)

    def _mutate_float(self, value, parameter_name: str, parameter_type: str):
        if self.is_discrete:
            return self._sample_discrete_float(parameter_name, parameter_type)
        return self._mutate_continuous_float(parameter_name, parameter_type, value)

    def _sample_discrete_float(self, parameter_name, parameter_type):
        possible_values = getattr(self.config, f'{parameter_name}_{parameter_type}_discrete')
        value = np.random.choice(possible_values, size=1)
        return value

    def _initialize_continuous_float(self, parameter_name, parameter_type):
        mean = getattr(self.config, f'{parameter_name}_{parameter_type}_init_mean')
        std = getattr(self.config, f'{parameter_name}_{parameter_type}_init_std')
        value = np.random.normal(loc=mean, scale=std)
        return value

    def _mutate_continuous_float(self, parameter_name, parameter_type, value):
        r = random.random()
        if r < self.mutate_rate:
            mutated_value = value + random.gauss(0.0, self.mutate_power)
            return mutated_value
        if r < self.replace_rate + self.mutate_rate:
            return self._initialize_float(parameter_name=parameter_name, parameter_type=parameter_type)
        return value

    def copy(self):
        raise NotImplementedError


class ConnectionGene(Gene):
    # crossover_attributes = ['mean', 'std']
    # mutation_attributes = ['mean']

    @staticmethod
    def from_dict(connection_gene_dict: dict):
        # key = ConnectionGene._get_tuple(tuple_str=connection_gene_dict['key'])
        key = tuple(connection_gene_dict['key'])
        connection_gene = ConnectionGene(key=key)
        connection_gene.set_mean(connection_gene_dict['_weight_mean'])
        connection_gene.set_std(connection_gene_dict['_weight_std'])
        return connection_gene

    @staticmethod
    def _get_tuple(tuple_str) -> tuple:
        tuple_str = tuple_str[1:-1].split(',')
        tuple_ = (int(tuple_str[0]), int(tuple_str[1]))
        return tuple_

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

    def random_initialization(self):
        weight_mean = self._initialize_float(parameter_name='weight', parameter_type='mean')
        self.set_mean(weight_mean)

        weight_std = self._initialize_float(parameter_name='weight', parameter_type='std')
        self.set_std(weight_std)
        return self

    def take_sample(self):
        return np.random.normal(loc=self.get_mean(), scale=self.get_std())

    def to_dict(self):

        dict_ = {'key': self.key,
                 'enabled': self.enabled,
                 '_weight_mean': self._weight_mean,
                 '_weight_std': self._weight_std}
        return dict_

    def copy(self):
        connection_copy = ConnectionGene(key=self.key)
        for attribute, value in self.__dict__.items():
            setattr(connection_copy, attribute, value)
        return connection_copy


class NodeGene(Gene):
    # crossover_attributes = ['mean', 'std']
    # mutation_attributes = ['mean']

    @staticmethod
    def from_dict(node_gene_dict: dict):
        key = int(node_gene_dict['key'])
        node_gene = NodeGene(key=key)
        node_gene.set_mean(node_gene_dict['_bias_mean'])
        node_gene.set_std(node_gene_dict['_bias_std'])
        return node_gene

    def __init__(self, key):
        super().__init__(key=key, type=NODE_TYPE)
        self.key = key
        self.config = get_configuration()

        self.activation = self.config.node_activation
        self.aggregation = self.config.node_aggregation

        self.bias_configuration = BiasConfig()

    def random_initialization(self):
        bias_mean = self._initialize_float(parameter_name='bias', parameter_type='mean')
        self.set_mean(bias_mean)

        bias_std = self._initialize_float(parameter_name='bias', parameter_type='std')
        self.set_std(bias_std)
        return self

    def take_sample(self):
        return np.random.normal(loc=self.get_mean(), scale=self.get_std())

    def to_dict(self):
        dict_ = {'key': self.key,
                 'activation': self.activation,
                 'aggregation': self.aggregation,
                 '_bias_mean': self._bias_mean,
                 '_bias_std': self._bias_std}
        return dict_

    def copy(self):
        node_copy = NodeGene(key=self.key)
        for attribute, value in self.__dict__.items():
            setattr(node_copy, attribute, value)
        return node_copy


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


def calculate_std_given_variance(var):
    return float(np.sqrt(var))


def calculate_log_var_given_variance(var):
    return float(np.log(var) - 1.0)


def calculate_variance_given_log_var(log_var):
    return float(np.exp(log_var + 1.0))


def calculate_variance_given_std(std):
    return float(np.square(std))
