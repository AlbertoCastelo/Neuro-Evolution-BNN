import copy
import random
import uuid
from itertools import count
import numpy as np
import jsons

from neat.configuration import get_configuration, write_json_file_from_dict, read_json_file_to_dict, BaseConfiguration
from neat.gene import NodeGene, ConnectionGene
from neat.utils import timeit


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
    def create_from_julia_dict(genome_dict: dict):
        config = get_configuration()
        genome = Genome(key=genome_dict["key"], id=None, genome_config=config)

        # reconstruct nodes and connections
        connection_genes_dict = genome_dict['connections']
        for key_str, connection_gene_dict in connection_genes_dict.items():
            connection_key = Genome._get_connection_key_from_key_str(key_str)
            connection_gene = ConnectionGene(key=connection_key)
            connection_gene.set_mean(connection_gene_dict['mean_weight'])
            connection_gene.set_std(connection_gene_dict['std_weight'])
            genome.connection_genes[connection_gene.key] = connection_gene

        node_genes_dict = genome_dict['nodes']
        for key_str, node_gene_dict in node_genes_dict.items():
            node_key = int(key_str)
            node_gene = NodeGene(key=node_key)
            node_gene.set_mean(node_gene_dict['mean_bias'])
            node_gene.set_std(node_gene_dict['std_bias'])
            genome.node_genes[node_gene.key] = node_gene

        genome.calculate_number_of_parameters()
        return genome

    @staticmethod
    def _get_connection_key_from_key_str(key_str):
        node_keys_str = key_str[1:-1].split(',')
        return (int(node_keys_str[0]), int(node_keys_str[1]))

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
    def create_from_file(filename, key=None):
        genome_dict = read_json_file_to_dict(filename)
        genome = Genome.from_dict(genome_dict)
        if key:
            genome.key = key
        return genome

    def __init__(self, key, id=None, genome_config=None):
        self.describe_with_parameters = False
        self.key = key
        self.id = str(uuid.uuid4()) if id is None else id

        self.genome_config = get_configuration() if genome_config is None else genome_config
        self.n_input = self.genome_config.n_input
        self.n_output = self.genome_config.n_output
        self.initial_nodes_sample = self.genome_config.initial_nodes_sample

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

    @timeit
    def copy(self):
        return Genome.from_dict(copy.deepcopy(self.to_dict()))

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
        node.set_mean(mean)
        node.set_std(std)
        self.node_genes[key] = node

    def add_connection(self, key, mean=None, std=None):
        connection = ConnectionGene(key=key)
        connection.set_mean(mean)
        connection.set_std(std)
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
        if self.genome_config.is_initial_fully_connected:
            # initialize fully connected network with no recurrent connections
            connections = self._compute_full_connections()
        else:
            # initialize network with only a few connections between input-output nodes
            connections = self._compute_random_connections(repetitions=min(self.initial_nodes_sample, len(self.input_nodes_keys)))

        for input_node, output_node in connections:
            key = (int(input_node), int(output_node))
            connection = ConnectionGene(key=key)
            connection.random_initialization()
            self.connection_genes[key] = connection

    def _compute_random_connections(self, repetitions):

        # each output node should contain at least 1 connection to input
        connections = []
        hidden_nodes = self._get_hidden_nodes()
        if hidden_nodes:
            for h in hidden_nodes:
                input_nodes_keys = np.random.choice(self.input_nodes_keys, size=repetitions)
                for input_id in input_nodes_keys:
                    connections.append((input_id, h))
            for h in hidden_nodes:
                for output_id in self.output_nodes_keys:
                    connections.append((h, output_id))
        else:

            for output_id in self.output_nodes_keys:
                input_nodes_keys = np.random.choice(self.input_nodes_keys, size=repetitions)
                for input_id in input_nodes_keys:
                    connection = (input_id, output_id)
                    if connection not in connections:
                        connections.append(connection)
        return connections

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

    def __eq__(self, other):
        if not self.check_same_architecture(other):
            return False

        # check Bias values
        for node_key, node_self in self.node_genes.items():
            node_other = other.node_genes[node_key]

            mean_difference = round(node_self.get_mean() - node_other.get_mean(), 4)
            std_difference = round(node_self.get_std() - node_other.get_std(), 4)
            if mean_difference != 0.0:
                print(f'Mean difference in Node')

            if std_difference != 0.0:
                print(f'Std difference in Node')

        # check Weight values
        for connection_key, connection_self in self.connection_genes.items():
            connection_other = other.connection_genes[connection_key]

            mean_difference = round(connection_self.get_mean() - connection_other.get_mean(), 4)
            std_difference = round(connection_self.get_std() - connection_other.get_std(), 4)
            if mean_difference != 0.0:
                print(f'Mean difference in Connection')

            if std_difference != 0.0:
                print(f'Std difference in Connection')

        return True

    def check_same_architecture(self, other):
        # check node keys
        nodes_self_but_other = set(self.node_genes.keys()) - set(other.node_genes.keys())
        nodes_other_but_self = set(other.node_genes.keys()) - set(self.node_genes.keys())
        if nodes_self_but_other != set():
            print(f'Nodes in self but not in other: {nodes_self_but_other}')
            return False
        if nodes_other_but_self != set():
            print(f'Nodes in other but not in self: {nodes_other_but_self}')
            return False

        # check connection keys
        connections_self_but_other = set(self.connection_genes.keys()) - set(other.connection_genes.keys())
        connections_other_but_self = set(other.connection_genes.keys()) - set(self.connection_genes.keys())
        if connections_self_but_other != set():
            print(f'Connections in self but not in other: {connections_self_but_other}')
            return False
        if connections_other_but_self != set():
            print(f'Connections in other but not in self: {connections_other_but_self}')
            return False
        return True
