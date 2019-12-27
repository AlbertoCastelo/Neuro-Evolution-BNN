import jsons

from experiments.file_utils import read_json_file_to_dict, write_json_file_from_dict


class ConfigError(Exception):
    def __init__(self, message):
        super().__init__(message)


class BaseConfiguration:
    pass

    def to_dict(self):
        return jsons.dump(self)


class DefaultConfiguration(BaseConfiguration):
    def __init__(self):
        # problem definition
        self.dataset = 'regression_example_1'
        self.problem_type = 'regression'

        self.is_discrete = True

        # Degrees of Freedom
        self.fix_std = True
        self.fix_architecture = True

        # logging levels
        self.log_network = False
        self.log_time = False
        self.log_population = False

        # evaluation parameters
        self.batch_size = 10000
        self.parallel_evaluation = True
        self.n_processes = None

        # evolution parameters
        self.pop_size = 150
        self.n_generations = 200

        # execution
        self.is_gpu = False

        # network parameters
        self.n_input = 1
        self.n_output = 1

        self.node_activation = 'sigmoid'
        self.node_aggregation = 'sum'

        # node genes configuration
        self.bias_mean_init_mean = 0.0
        self.bias_mean_init_std = 1.0
        self.bias_std_init_mean = 1.0
        self.bias_std_init_std = 0.0

        self.bias_mean_max_value = 30.0
        self.bias_mean_min_value = -30.0
        self.bias_std_max_value = 2
        self.bias_std_min_value = 0

        self.bias_mutate_power = 0.5
        self.bias_mutate_rate = 0.7
        self.bias_replace_rate = 0.1

        self.n_initial_hidden_neurons = 5

        # TODO: THIS IS REDUNDANT AND INNECESARY BECAUSE IT DOES NOT CHANGE
        self.response_init_mean = 1.0
        self.response_init_std = 0.0
        self.response_max_value = 30.0
        self.response_min_value = -30.0
        self.response_mutate_power = 0.0
        self.response_mutate_rate = 0.0
        self.response_replace_rate = 0.00

        # Connection Genes Configuration
        self.weight_mean_init_mean = 0
        self.weight_mean_init_std = 1
        self.weight_mean_max_value = 10
        self.weight_mean_min_value = -10

        # it starts with fixed std
        self.weight_std_init_mean = 1
        self.weight_std_init_std = 0
        self.weight_std_max_value = 2
        self.weight_std_min_value = 0

        # PRIORS
        self.weight_mean_prior = 0.0
        self.weight_std_prior = 1.0
        self.bias_mean_prior = 0.0
        self.bias_std_prior = 1.0

        # loss weighting factor
        self.beta_type = 'other'

        self.n_samples = 100

        # speciation
        self.compatibility_threshold = 3.0
        self.compatibility_weight_coefficient = 0.5
        self.compatibility_disjoint_coefficient = 1.0

        # stagnation
        self.species_fitness_function = 'max'
        self.species_elitism = 2
        self.max_stagnation = 20

        # reproduction
        self.elitism = 2
        self.survival_threshold = 0.2
        self.min_species_size = 2

        # mutation
        self.single_structural_mutation = False
        self.mutate_power = 0.5
        self.mutate_rate = 0.8
        self.replace_rate = 0.1

    def _get_configuration(self):
        configuration = {}
        for attr_name in dir(self):
            if attr_name[0] == '_':
                continue
            configuration[attr_name] = getattr(self, attr_name)
        return configuration

    def _write_to_json(self, filename):
        config = self._get_configuration()
        write_json_file_from_dict(data=config, filename=filename)


class _Configuration:
    _instance = None

    def __init__(self, filename):
        if filename is None:
            configuration = DefaultConfiguration()._get_configuration()
        else:
            configuration = self.read_configuration_from_file(filename)

        assert isinstance(configuration, dict)
        config = BaseConfiguration()
        for key in configuration:
            setattr(config, key, configuration[key])

        # self.process_configuration()

        _Configuration._instance = config

    # def process_configuration(self):
    #     self.is_classification = True
    #     if self.problem_type == 'regression':
    #         self.is_classification = False
    #     else:
    #         raise ConfigError(f'Problem Type is incorrect: {self.problem_type}')

    def read_configuration_from_file(self, filename):
        configuration = read_json_file_to_dict(filename)
        return configuration


def get_configuration(filename=None):
    if _Configuration._instance is None or filename is not None:
        _Configuration(filename=filename)
    return _Configuration._instance
