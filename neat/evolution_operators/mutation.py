from neat.configuration import get_configuration
from neat.genome import Genome


class Mutation:

    def __init__(self):
        self.config = get_configuration()
        self.single_structural_mutation = self.config.single_structural_mutation
        self.mutate_rate = self.config.mutate_rate
        self.mutate_power = self.config.mutate_power
        self.replace_rate = self.config.replace_rate

        # self.node_add_prob = self.config.node_add_prob
        # self.node_delete_prob = self.config.node_delete_prob
        # self.conn_add_prob = self.config.conn_add_prob
        # self.conn_delete_prob = self.config.conn_delete_prob

    def mutate(self, genome: Genome):
        # if self.single_structural_mutation:
        #     div = max(1, (self.node_add_prob + self.node_delete_prob +
        #                   self.conn_add_prob + self.conn_delete_prob))
        #     r = random.random()
        #     if r < (self.node_add_prob/div):
        #         genome = self.mutate_add_node(genome)
        #     elif r < ((self.node_add_prob + self.node_delete_prob)/div):
        #         genome = self.mutate_delete_node(genome)
        #     elif r < ((self.node_add_prob + self.node_delete_prob +
        #                self.conn_add_prob)/div):
        #         genome = self.mutate_add_connection(genome)
        #     elif r < ((self.node_add_prob + self.node_delete_prob +
        #                self.conn_add_prob + self.conn_delete_prob)/div):
        #         genome = self.mutate_delete_connection(genome)
        # else:
        #     if random.random() < self.node_add_prob:
        #         genome = self.mutate_add_node(genome)
        #
        #     if random.random() < self.node_delete_prob:
        #         genome = self.mutate_delete_node(genome)
        #
        #     if random.random() < self.conn_add_prob:
        #         genome = self.mutate_add_connection(genome)
        #
        #     if random.random() < self.conn_delete_prob:
        #         genome = self.mutate_delete_connection(genome)

        # Mutate connection genes.
        for key in genome.connection_genes.keys():
            genome.connection_genes[key].mutate()
        # Mutate node genes (bias, response, etc.).
        for key in genome.node_genes.keys():
            genome.node_genes[key].mutate()
        return genome

    def mutate_add_node(self, genome):
        raise
        return genome

    def mutate_delete_node(self, genome):
        pass

    def mutate_add_connection(self, genome):
        pass

    def mutate_delete_connection(self, genome: Genome):
        pass
