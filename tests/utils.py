from neat.gene import NodeGene, ConnectionGene
from neat.genome import Genome


def generate_genome_with_hidden_units():

    # output nodes
    node_0 = NodeGene(key=0)
    node_0.random_initialization()

    # hidden nodes
    node_1 = NodeGene(key=1)
    node_1.random_initialization()

    node_2 = NodeGene(key=2)
    node_2.random_initialization()

    node_3 = NodeGene(key=3)
    node_3.random_initialization()

    node_genes = {0: node_0, 1: node_1, 2: node_2, 3: node_3}

    connection_genes = {}
    connection_genes = add_connection(connection_genes, key=(-1, 1))
    connection_genes = add_connection(connection_genes, key=(-1, 2))
    connection_genes = add_connection(connection_genes, key=(-1, 3))
    connection_genes = add_connection(connection_genes, key=(3, 0))
    connection_genes = add_connection(connection_genes, key=(2, 0))
    connection_genes = add_connection(connection_genes, key=(1, 0))

    genome = Genome(key=1)
    genome.node_genes = node_genes
    genome.connection_genes = connection_genes
    return genome


def add_connection(connection_genes, key):
    connection_i = ConnectionGene(key=key)
    connection_i.random_initialization()
    connection_genes[key] = connection_i
    return connection_genes
