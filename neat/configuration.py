class BaseConfiguration:
    pass

class DefaultConfiguration:
    def __init__(self):
        # evolution parameters
        self.pop_size = 150
        self.n_generations = 200

        # network parameters
        self.n_input = 5
        self.n_output = 1

        self.activation = 'sigmoid'
        self.node_aggregation = 'sum'

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
