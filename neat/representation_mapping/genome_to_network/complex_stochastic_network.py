import torch
from torch import nn
from torch.distributions import Normal

from neat.evolution_operators.mutation import _get_connections_per_node
from neat.genome import Genome
from neat.representation_mapping.genome_to_network.layers import StochasticLinearParameters, StochasticLinear, \
    ComplexStochasticLinear
from neat.representation_mapping.genome_to_network.stochastic_network import _is_next_layer_input, \
    _filter_nodes_without_input_connection
from neat.representation_mapping.genome_to_network.utils import get_activation


class ComplexStochasticNetwork(nn.Module):
    def __init__(self, genome: Genome):
        super(ComplexStochasticNetwork, self).__init__()

        self.n_output = genome.n_output
        self.n_input = genome.n_input
        self.nodes = genome.node_genes
        self.connections = genome.connection_genes

        self.activation = get_activation()
        self.layers = transform_genome_to_layers(genome=genome)
        self.n_layers = len(self.layers)
        self._set_network_layers(layers=self.layers)
        self._cache = {}

    def forward(self, x):
        kl_qw_pw = 0.0
        start_index = self.n_layers - 1

        for i in range(start_index, -1, -1):
            # cache needed values
            for index_to_cache in self.layers[i].indeces_of_nodes_to_cache:
                # self._cache[(i, index_to_cache)] = x[:, index_to_cache]
                self._cache[(i, index_to_cache)] = x.index_select(1, torch.LongTensor((index_to_cache,)))
            # append needed values
            chunks = [x]
            for index_needed in self.layers[i].indeces_of_needed_nodes:
                chunks.append(self._cache[index_needed])
            x = torch.cat(chunks, 1)

            x = getattr(self, f'layer_{i}')(x)
            if i > 0:
                # x = getattr(self, f'activation_{i}')(x)
                pass


        return x, kl_qw_pw

    def _set_network_layers(self, layers: dict):
        for layer_key in layers:
            layer = layers[layer_key]

            parameters = StochasticLinearParameters.create(qw_mean=layer.weight_mean,
                                                           qw_logvar=layer.weight_log_var,
                                                           qb_mean=layer.bias_mean,
                                                           qb_logvar=layer.bias_log_var)

            layer = ComplexStochasticLinear(in_features=layer.n_input,
                                            out_features=layer.n_output,
                                            parameters=parameters)

            setattr(self, f'layer_{layer_key}', layer)
            setattr(self, f'activation_{layer_key}', self.activation)


def get_nodes_per_depth_level(links: list):
    max_graph_depth_per_node = calculate_max_graph_depth_per_node(links=links)

    # get nodes at each depth level
    nodes_per_depth_level = {}
    for node_key, depth in max_graph_depth_per_node.items():
        if depth in nodes_per_depth_level:
            nodes_per_depth_level[depth].append(node_key)
        else:
            nodes_per_depth_level[depth] = [node_key]

    return nodes_per_depth_level


def calculate_max_graph_depth_per_node(links: list):
    input_nodes = list(zip(*links))[0]
    output_nodes = list(zip(*links))[1]
    all_nodes = list(set(input_nodes).union(set(output_nodes)))

    connections_per_node = _get_connections_per_node(connections=links, inverse_order=True)

    # get depths_per_node
    max_graph_depth_per_node = {}
    for node_key in all_nodes:
        depth = _get_depth_per_node(node_key=node_key, connections_per_node=connections_per_node,
                                    max_graph_depth_per_node=max_graph_depth_per_node)
        max_graph_depth_per_node[node_key] = depth

    return max_graph_depth_per_node


def _get_depth_per_node(node_key: int, connections_per_node: dict, max_graph_depth_per_node: dict):
    if node_key not in connections_per_node:
        return 0

    # all parent nodes are negative
    if _is_next_layer_input(layer_node_keys=connections_per_node[node_key]):
        return 1

    max_depth = 0
    for parent_node_key in connections_per_node[node_key]:
        if parent_node_key in max_graph_depth_per_node:
            max_depth_candidate = max_graph_depth_per_node[parent_node_key] + 1
        elif parent_node_key < 0:
            max_depth_candidate = 0
        else:
            max_depth_candidate = _get_depth_per_node(node_key=parent_node_key,
                                                      connections_per_node=connections_per_node,
                                                      max_graph_depth_per_node=max_graph_depth_per_node) + 1

        if max_depth_candidate > max_depth:
            max_depth = max_depth_candidate
    return max_depth


def transform_genome_to_layers(genome: Genome) -> dict:
    layers = dict()

    output_node_keys = genome.get_output_nodes_keys()
    nodes = genome.node_genes
    connections = genome.connection_genes
    layer_node_keys = output_node_keys

    nodes_per_depth_level = get_nodes_per_depth_level(links=list(connections.keys()))

    layer_counter = 0
    is_not_finished = True
    while is_not_finished:
        layer = LayerBuilder(nodes=nodes,
                             connections=connections,
                             layer_node_keys=layer_node_keys,
                             nodes_per_depth_level=nodes_per_depth_level,
                             layer_counter=layer_counter)\
            .create()\
            .get_layer()

        layer_node_keys = layer.get_original_input_keys()
        layers[layer_counter] = layer
        layer_counter += 1
        if _is_next_layer_input(layer_node_keys):
            is_not_finished = False
    return layers


class Layer:
    def __init__(self, key, n_input, n_output):
        self.key = key
        self.n_input = n_input
        self.n_output = n_output
        self.input_keys = None
        self.external_input_keys = None
        self.original_input_keys = None
        self.output_keys = None

        self.indeces_of_nodes_to_cache = []
        self.indeces_of_needed_nodes = []

        # parameters
        self.bias_mean = None
        self.bias_log_var = None
        self.weight_mean = None
        self.weight_log_var = None

    def update_input_keys(self):
        if self.original_input_keys is not None and self.input_keys is not None:
            self.external_input_keys = list(set(self.input_keys) - set(self.original_input_keys))
        else:
            raise ValueError

    def get_original_input_keys(self):
        return self.original_input_keys

    def validate(self):
        # validate shapes
        pass


class LayerBuilder:

    def __init__(self, nodes, connections, layer_node_keys, nodes_per_depth_level, layer_counter):
        '''
            :param nodes: dictionary with all the nodes in the genome
            :param connections: dictionary with all the connections of the genome
            :param layer_node_keys: a list of keys of the nodes in a layer
        '''
        self.nodes = nodes
        self.connections = connections
        self.layer_node_keys = layer_node_keys
        self.nodes_per_depth_level = nodes_per_depth_level
        self.layer_counter = layer_counter

        self.layer = None

    def create(self):
        layer_node_keys = _filter_nodes_without_input_connection(node_keys=self.layer_node_keys,
                                                                 connections=self.connections)
        layer_node_keys.sort()
        n_output = len(layer_node_keys)

        layer_connections = dict()
        input_node_keys = set({})
        for connection_key, connection in self.connections.items():
            input_node_key, output_node_key = connection_key
            if output_node_key in layer_node_keys:
                input_node_keys = input_node_keys.union({input_node_key})
                if connection.enabled:
                    layer_connections[connection_key] = self.connections[connection_key]

        input_node_keys = list(input_node_keys)
        original_input_keys = self._get_original_input_keys()
        n_input = len(input_node_keys)

        self.layer = Layer(key=self.layer_counter, n_input=n_input, n_output=n_output)
        self.layer.input_keys = input_node_keys
        self.layer.original_input_keys = original_input_keys
        self.layer.update_input_keys()
        self.layer.output_keys = layer_node_keys

        # set parameters
        self.layer.weight_mean, self.layer.weight_log_var = \
            self._build_weight_tensors(input_node_keys=input_node_keys,
                                       layer_connections=layer_connections,
                                       layer_node_keys=layer_node_keys,
                                       n_input=n_input,
                                       n_output=n_output)

        self.layer.bias_mean, self.layer.bias_log_var = self._build_bias_tensors(layer_node_keys, self.nodes)

        self.layer.validate()
        return self

    def get_layer(self):
        return self.layer

    @staticmethod
    def _build_weight_tensors(input_node_keys, layer_connections, layer_node_keys, n_input, n_output):
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
        return weight_mean, weight_log_var

    @staticmethod
    def _build_bias_tensors(layer_node_keys, nodes):
        bias_mean_values = [nodes[key].get_mean() for key in layer_node_keys]
        bias_mean = torch.tensor(bias_mean_values)

        bias_log_var_values = [nodes[key].get_log_var() for key in layer_node_keys]
        bias_log_var = torch.tensor(bias_log_var_values)
        return bias_mean, bias_log_var

    def _get_original_input_keys(self):
        distances = list(self.nodes_per_depth_level.keys())
        distances.sort(reverse=True)
        # distances.reverse()
        distance_key = distances[self.layer_counter + 1]
        return self.nodes_per_depth_level[distance_key]

# def build_layer_parameters(nodes, connections, layer_node_keys, max_graph_depth_per_node) -> dict:
#     '''
#     :param nodes: dictionary with all the nodes in the genome
#     :param connections: dictionary with all the connections of the genome
#     :param layer_node_keys: a list of keys of the nodes in a layer
#     :return: dictionary with all parameters associated with a layer
#     '''
#     layer = dict()
#
#     layer_node_keys = _filter_nodes_without_input_connection(node_keys=layer_node_keys, connections=connections)
#     layer_node_keys.sort()
#     n_output = len(layer_node_keys)
#
#     layer_connections = dict()
#     input_node_keys = set({})
#     for connection_key, connection in connections.items():
#         input_node_key, output_node_key = connection_key
#         if output_node_key in layer_node_keys:
#             input_node_keys = input_node_keys.union({input_node_key})
#             if connection.enabled:
#                 layer_connections[connection_key] = connections[connection_key]
#
#     input_node_keys = list(input_node_keys)
#     n_input = len(input_node_keys)
#
#     weight_mean, weight_log_var = _build_weight_tensors(input_node_keys=input_node_keys,
#                                                         layer_connections=layer_connections,
#                                                         layer_node_keys=layer_node_keys,
#                                                         n_input=n_input,
#                                                         n_output=n_output)
#
#     bias_mean, bias_log_var = _build_bias_tensors(layer_node_keys, nodes)
#
#     # set parameters
#     layer['n_input'] = n_input
#     layer['n_output'] = n_output
#     layer['input_keys'] = input_node_keys
#     layer['output_keys'] = layer_node_keys
#     layer['bias_mean'] = bias_mean
#     layer['bias_log_var'] = bias_log_var
#     layer['weight_mean'] = weight_mean
#     layer['weight_log_var'] = weight_log_var
#     return layer



