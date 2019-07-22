class ConfigError(Exception):
    def __init__(self, message):
        super().__init__(message)


class BaseConfiguration:
    pass


class DefaultConfiguration:
    def __init__(self):
        # problem definition
        self.dataset_name = 'regression_example_1'
        self.problem_type = 'regression'

        # evaluation parameters
        self.batch_size = 10000

        # evolution parameters
        self.pop_size = 150
        self.n_generations = 200

        # execution
        self.is_gpu = False

        # network parameters
        self.n_input = 5
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

        self.initial_hidden = False

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
        self.beta = 'Standard'

    def get_configuration(self):
        configuration = {}
        for attr_name in dir(self):
            if '__' in attr_name:
                continue
            configuration[attr_name] = getattr(self, attr_name)
        return configuration


class _Configuration:
    _instance = None

    def __init__(self, filename):
        configuration = self.read_configuration_from_file(filename)
        if configuration is None:
            configuration = DefaultConfiguration().get_configuration()

        assert isinstance(configuration, dict)
        config = BaseConfiguration()
        for key in configuration:
            setattr(config, key, configuration[key])

        _Configuration._instance = config

    def read_configuration_from_file(self, filename):
        return None


def get_configuration(filename=None):
    if _Configuration._instance is None:
        _Configuration(filename=filename)
    return _Configuration._instance

