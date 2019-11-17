import uuid
from itertools import count

import jsons

from neat.configuration import get_configuration, write_json_file_from_dict, read_json_file_to_dict, BaseConfiguration
from neat.gene import NodeGene, ConnectionGene


class GenomeSample:
    def __init__(self, key, n_output, n_input, node_genes, connection_genes):
        self.key = key
        self.n_output = n_output
        self.n_input = n_input
        self.node_genes = node_genes
        self.connection_genes = connection_genes
        self.fitness = None


class Genome:
    @staticmethod
    def from_dict(genome_dict: dict):
        genome_config_dict = genome_dict['config']
        genome_config = jsons.load(genome_config_dict, BaseConfiguration)
        genome = Genome(key=genome_dict['key'], id=genome_dict['id'], genome_config=genome_config)

        # reconstruct nodes and connections
        connection_genes_dict = genome_dict['connection_genes']
        for key, connection_gene_dict in connection_genes_dict.items():
            connection_gene = ConnectionGene.from_dict(connection_gene_dict=connection_gene_dict)
            genome.connection_genes[connection_gene.key] = connection_gene

        node_genes_dict = genome_dict['node_genes']
        for key, node_gene_dict in node_genes_dict.items():
            node_gene = NodeGene.from_dict(node_gene_dict=node_gene_dict)
            genome.node_genes[node_gene.key] = node_gene

        genome.n_weight_parameters = genome_dict['n_weight_parameters']
        genome.n_bias_parameters = genome_dict['n_bias_parameters']
        genome.node_counter = count(max(list(genome.node_genes.keys())) + 1)
        return genome

    @staticmethod
    def create_from_file(filename):
        genome_dict = read_json_file_to_dict(filename)
        return Genome.from_dict(genome_dict)

    def __init__(self, key, id=None, genome_config=None):
        self.describe_with_parameters = False
        self.key = key
        self.id = str(uuid.uuid4()) if id is None else id

        self.genome_config = get_configuration() if genome_config is None else genome_config
        self.n_input = self.genome_config.n_input
        self.n_output = self.genome_config.n_output

        self.output_nodes_keys = self.get_output_nodes_keys()
        self.input_nodes_keys = self.get_input_nodes_keys()

        self.connection_genes = {}
        self.node_genes = {}

        self.n_weight_parameters = None
        self.n_bias_parameters = None

        self.node_counter = None

        self.fitness = None

    def to_dict(self):
        dict_ = {'key': self.key,
                 'id': self.id,
                 'n_input': self.n_input,
                 'n_output': self.n_output,
                 'output_nodes_keys': self.output_nodes_keys,
                 'input_nodes_keys': self.input_nodes_keys,
                 'connection_genes': {},
                 'node_genes': {},
                 'n_bias_parameters': self.n_bias_parameters,
                 'n_weight_parameters': self.n_weight_parameters,
                 'config': self.genome_config.to_dict()}

        for key, connection_gene in self.connection_genes.items():
            dict_['connection_genes'][str(key)] = connection_gene.to_dict()

        for key, node_gene in self.node_genes.items():
            dict_['node_genes'][str(key)] = node_gene.to_dict()

        return dict_

    def save_genome(self, filename=None):
        filename = ''.join([str(uuid.uuid4()), '.json']) if filename is None else filename

        write_json_file_from_dict(data=self.to_dict(), filename=filename)

    def create_random_genome(self):
        self._initialize_output_nodes()

        # initialize hidden units
        self._initialize_hidden_nodes()

        # initialize connections
        self._initialize_connections()
        return self

    def add_node(self, key, mean=None, std=None):
        node = NodeGene(key=key)
        node.bias_mean = mean
        node.bias_std = std
        self.node_genes[key] = node

    def add_connection(self, key, mean=None, std=None):
        connection = ConnectionGene(key=key)
        connection.weight_mean = mean
        connection.weight_std = std
        self.connection_genes[key] = connection

    def get_genome_sample(self):
        '''
        Takes a sample of the network to be analyzed. This translates in taking a sample of each connection distribution
        '''
        connection_genes_sample = {}
        for key, connection_gene in self.connection_genes.items():
            connection_genes_sample[key] = connection_gene.take_sample()

        node_genes_sample = {}
        for key, node_gene in self.node_genes.items():
            node_genes_sample[key] = node_gene.take_sample()

        return GenomeSample(key=self.key, n_input=self.n_input, n_output=self.n_output,
                            node_genes=node_genes_sample,
                            connection_genes=connection_genes_sample)

    def get_new_node_key(self):
        if self.node_counter is None:
            self.node_counter = count(max(list(self.node_genes.keys())) + 1)

        new_key = next(self.node_counter)
        assert new_key not in self.node_genes
        return new_key

    def calculate_number_of_parameters(self):
        self.n_weight_parameters = 2 * len(self.connection_genes)
        self.n_bias_parameters = 2 * len(self.node_genes)
        return self.n_weight_parameters + self.n_bias_parameters

    def get_input_nodes_keys(self):
        # input nodes only contain keys (they cannot be evolved)
        return list(range(-1, -self.n_input-1, -1))

    def get_output_nodes_keys(self):
        return list(range(0, self.n_output))

    def get_graph(self):
        connections = list(self.connection_genes.keys())
        return str(connections)

    def _initialize_connections(self):
        # initialize fully connected network with no recurrent connections
        for input_node, output_node in self._compute_full_connections():
            key = (input_node, output_node)
            connection = ConnectionGene(key=key)
            connection.random_initialization()
            self.connection_genes[key] = connection

    def _compute_full_connections(self):
        connections = []
        hidden_nodes = self._get_hidden_nodes()
        if hidden_nodes:
            for input_id in self.input_nodes_keys:
                for h in hidden_nodes:
                    connections.append((input_id, h))
            for h in hidden_nodes:
                for output_id in self.output_nodes_keys:
                    connections.append((h, output_id))
            return connections
        for input_id in self.input_nodes_keys:
            for output_id in self.output_nodes_keys:
                connections.append((input_id, output_id))
        # TODO: include recurrent connections if configured
        return connections

    def _initialize_hidden_nodes(self):
        '''
        hidden nodes have keys starting at n_outputs
        '''
        n_hidden = self.genome_config.n_initial_hidden_neurons

        for key in range(self.n_output, self.n_output + n_hidden):
            node = NodeGene(key=key)
            node.random_initialization()
            self.node_genes[key] = node

    def _initialize_output_nodes(self):
        for key in self.output_nodes_keys:
            node = NodeGene(key=key)
            node.random_initialization()
            self.node_genes[key] = node

    def _get_hidden_nodes(self):
        return list(self.node_genes.keys())[self.n_output:]

    def __str__(self):
        general_data = ''.join([f'Total number of Parameters: {self.calculate_number_of_parameters()}\n',
                                f'    N-Bias-Parameters: {self.n_bias_parameters}\n',
                                f'    N-Weight-Parameters: {self.n_weight_parameters}\n',
                                f'Fitness: {self.fitness}\n'])

        if self.describe_with_parameters:
            bias_data = []
            bias_data.append(''.join(['Node Key | Mean  | Std \n']))
            for key, node in self.node_genes.items():
                bias_data.append(''.join([str(key), ':     ', str(round(node.get_mean(), 4)), '  |  ',
                                          str(round(node.get_std(), 4)), ' \n']))
            bias_str = ''.join(bias_data)

            weights_data = []
            weights_data.append(''.join(['Connection Key | Mean  | Std \n']))
            for key, connection in self.connection_genes.items():
                weights_data.append(''.join([str(key), ': ', str(round(connection.get_mean(), 4)), '  |  ',
                                             str(round(connection.get_std(), 4)), ' \n']))
            weight_str = ''.join(weights_data)

            general_data = ''.join([general_data, '\n', bias_str, '\n', weight_str])
        return general_data
