from itertools import product

from neat.gene import NodeGene, ConnectionGene
from neat.genome import Genome


def generate_genome_without_hidden_units():
    # output nodes
    node_genes = {}
    node_genes = add_node(node_genes, key=0)

    connection_genes = {}
    connection_genes = add_connection(connection_genes, key=(-1, 0))
    connection_genes = add_connection(connection_genes, key=(-2, 0))

    genome = Genome(key=1)
    genome.node_genes = node_genes
    genome.connection_genes = connection_genes
    return genome


def generate_genome_with_hidden_units(n_input, n_output):
    N_HIDDEN = 3
    # nodes
    node_genes = {}
    for i in range(n_output + N_HIDDEN):
        node_genes = add_node(node_genes, key=i)

    # connections
    # input to hidden
    connection_genes = {}
    input_hidden_tuples = list(product(list(range(-1, -n_input-1, -1)),
                                       list(range(n_output, n_output+N_HIDDEN))))
    for tuple_ in input_hidden_tuples:
        connection_genes = add_connection(connection_genes, key=tuple_)

    # hidden to output
    hidden_output_tuples = list(product(list(range(n_output, n_output + N_HIDDEN)),
                                        list(range(0, n_output))))
    for tuple_ in hidden_output_tuples:
        connection_genes = add_connection(connection_genes, key=tuple_)

    # initialize genome
    genome = Genome(key=1)
    genome.node_genes = node_genes
    genome.connection_genes = connection_genes
    return genome


def add_node(node_genes, key):
    node_i = NodeGene(key=key)
    node_i.random_initialization()
    node_genes[key] = node_i
    return node_genes


def add_connection(connection_genes, key):
    connection_i = ConnectionGene(key=key)
    connection_i.random_initialization()
    connection_genes[key] = connection_i
    return connection_genes
