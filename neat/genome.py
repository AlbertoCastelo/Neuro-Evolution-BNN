import uuid

from neat.configuration import get_configuration

NODE_TYPE = 'node'
CONNECTION_TYPE = 'connection'


class Genome:

    def __init__(self, key, ):
        self.key = key
        self.id = uuid.uuid4()

        self.genome_config = get_configuration()
        self.n_input = self.genome_config.n_input
        self.n_output = self.genome_config.n_output

        self.connection_genes = {}
        self.node_genes = {}

        self.fitness = None

    def create_random_genome(self):
        for key in self.get_output_node_keys():
            self.node_genes[key] = NodeGene(key=key)

    def get_output_node_keys(self):
        '''
        Output nodes have keys: 0, ..
        '''
        return range(self.n_output)


class Gene:

    def __init__(self, key, type):
        self.key = key
        self.type = type


class ConnectionGene(Gene):

    def __init__(self, key):
        super().__init__(key=key, type=CONNECTION_TYPE)
        self.key = key


class NodeGene(Gene):

    def __init__(self, key):
        super().__init__(key=key, type=NODE_TYPE)
        self.key = key
        self.config = get_configuration()
        
        self.activation = self.config.node_activation
        self.aggregation = self.config.node_aggregation
        
        self.bias_configuration = self.get_bias_configuration()

    def get_bias_configuration(self):
