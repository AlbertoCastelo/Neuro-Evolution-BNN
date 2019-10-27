import torch
from torch import nn
from torch.distributions import Normal
from neat.genome import Genome
from neat.representation_mapping.genome_to_network.stochastic_network import _is_next_layer_input, \
    _filter_nodes_without_input_connection


class ComplexStochasticNetwork(nn.Module):
    def __init__(self, genome: Genome, n_samples: int):
        super(ComplexStochasticNetwork, self).__init__()
        self.genome = genome
        self.n_samples = n_samples

        self.n_output = genome.n_output
        self.n_input = genome.n_input
        self.nodes = genome.node_genes
        self.connections = genome.connection_genes


def transform_genome_to_layers(genome: Genome) -> dict:
    layers = dict()

    output_node_keys = genome.get_output_nodes_keys()
    nodes = genome.node_genes
    connections = genome.connection_genes

    layer_node_keys = output_node_keys

    layer_counter = 0
    is_not_finished = True
    while is_not_finished:
        layer = build_layer_parameters(nodes=nodes,
                                       connections=connections,
                                       layer_node_keys=layer_node_keys)
        layer_node_keys = layer['input_keys']
        layers[layer_counter] = layer
        layer_counter += 1
        if _is_next_layer_input(layer_node_keys):
            is_not_finished = False
    return layers


def build_layer_parameters(nodes, connections, layer_node_keys):
    layer = dict()

    layer_node_keys = _filter_nodes_without_input_connection(node_keys=layer_node_keys, connections=connections)
    n_output = len(layer_node_keys)

    # get bias
    layer_node_keys.sort()
    bias_mean_values = [nodes[key].get_mean() for key in layer_node_keys]
    bias_mean = torch.tensor(bias_mean_values)

    bias_log_var_values = [nodes[key].get_log_var() for key in layer_node_keys]
    bias_log_var = torch.tensor(bias_log_var_values)

    layer_connections = dict()
    input_node_keys = set({})
    for connection_key, connection in connections.items():
        input_node_key, output_node_key = connection_key
        if output_node_key in layer_node_keys:
            input_node_keys = input_node_keys.union({input_node_key})
            if connection.enabled:
                layer_connections[connection_key] = connections[connection_key]

    input_node_keys = list(input_node_keys)
    n_input = len(input_node_keys)

    key_index_mapping_input = dict()
    for input_index, input_key in enumerate(input_node_keys):
        key_index_mapping_input[input_key] = input_index

    key_index_mapping_output = dict()
    for output_index, output_key in enumerate(layer_node_keys):
        key_index_mapping_output[output_key] = output_index

    weight_mean = torch.zeros([n_output, n_input])
    weight_log_var = torch.zeros([n_output, n_input])
    for connection_key in layer_connections:
        input_node_key, output_node_key = connection_key
        weight_mean[key_index_mapping_output[output_node_key], key_index_mapping_input[input_node_key]] = \
            float(layer_connections[connection_key].get_mean())
        weight_log_var[key_index_mapping_output[output_node_key], key_index_mapping_input[input_node_key]] = \
            float(layer_connections[connection_key].get_log_var())

    layer['n_input'] = n_input
    layer['n_output'] = n_output
    layer['input_keys'] = input_node_keys
    layer['output_keys'] = layer_node_keys
    layer['bias_mean'] = bias_mean
    layer['bias_log_var'] = bias_log_var
    layer['weight_mean'] = weight_mean
    layer['weight_log_var'] = weight_log_var
    return layer
