import uuid

from neat.configuration import get_configuration
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

    def __init__(self, key):
        self.key = key
        self.id = uuid.uuid4()

        self.genome_config = get_configuration()
        self.n_input = self.genome_config.n_input
        self.n_output = self.genome_config.n_output

        self.connection_genes = {}
        self.node_genes = {}
        self.input_keys = self._initialize_input_nodes()
        self.fitness = None

    def create_random_genome(self):
        self._initialize_output_nodes()

        # initialize hidden units
        self._initialize_hidden_nodes()

        # initialize connections
        self._initialize_connections()

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
            for input_id in self.input_keys:
                for h in hidden_nodes:
                    connections.append((input_id, h))
            for h in hidden_nodes:
                for output_id in self._get_output_node_keys():
                    connections.append((h, output_id))
        for input_id in self.input_keys:
            for output_id in self._get_output_node_keys():
                connections.append((input_id, output_id))
        # TODO: include recurrent connections if configured
        return connections

    def _initialize_hidden_nodes(self):
        '''
        hidden nodes have keys starting at n_outputs
        '''
        if self.genome_config.initial_hidden:
            pass

    def _initialize_output_nodes(self):
        for key in self._get_output_node_keys():
            node = NodeGene(key=key)
            node.random_initialization()
            self.node_genes[key] = node

    def _get_output_node_keys(self):
        '''
        Output nodes have keys: 0, ..
        '''
        return range(self.n_output)

    def _get_hidden_nodes(self):
        return list(self.node_genes.keys())[self.n_output:]

    def _initialize_input_nodes(self):
        # input nodes only contain keys (they cannot be evolved)
        return list(range(-1, -self.n_input-1, -1))



