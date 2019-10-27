import torch
from torch import nn
from torch.distributions import Normal

from neat.genome import Genome
from neat.representation_mapping.genome_to_network.layers import StochasticLinear, StochasticLinearParameters
from neat.representation_mapping.genome_to_network.utils import get_activation


class StochasticNetwork(nn.Module):
    def __init__(self, genome: Genome, n_samples):
        super(StochasticNetwork, self).__init__()
        self.n_samples = n_samples

        self.n_output = genome.n_output
        self.n_input = genome.n_input
        self.nodes = genome.node_genes
        self.connections = genome.connection_genes

        self.activation = get_activation()
        layers_dict = _transform_genome_to_layers(nodes=self.nodes,
                                                  connections=self.connections,
                                                  n_output=self.n_output)
        self.n_layers = len(layers_dict)
        self._set_network_layers(layers=layers_dict)

    def forward(self, x):
        kl_qw_pw = 0.0
        start_index = self.n_layers - 1
        for i in range(start_index, -1, -1):
            # TODO: remove kl_estimation from layer because one parameter could show un in several layers...
            x, kl_layer = getattr(self, f'layer_{i}')(x)
            kl_qw_pw += kl_layer
            if i > 0:
                x = getattr(self, f'activation_{i}')(x)

        return x, kl_qw_pw

    def _set_network_layers(self, layers: dict):
        for layer_key in layers:
            layer_dict = layers[layer_key]

            parameters = StochasticLinearParameters.create(qw_mean=layer_dict['weight_mean'],
                                                           qw_logvar=layer_dict['weight_log_var'],
                                                           qb_mean=layer_dict['bias_mean'],
                                                           qb_logvar=layer_dict['bias_log_var'])

            layer = StochasticLinear(in_features=layer_dict['n_input'],
                                     out_features=layer_dict['n_output'],
                                     parameters=parameters,
                                     n_samples=self.n_samples)

            setattr(self, f'layer_{layer_key}', layer)
            setattr(self, f'activation_{layer_key}', self.activation)


class StochasticNetworkOld(nn.Module):
    def __init__(self, genome: Genome):
        super(StochasticNetworkOld, self).__init__()
        self.n_output = genome.n_output
        self.n_input = genome.n_input
        self.nodes = genome.node_genes
        self.connections = genome.connection_genes

        self.activation = get_activation()
        layers_dict = _transform_genome_to_layers(nodes=self.nodes,
                                                  connections=self.connections,
                                                  n_output=self.n_output)
        self.n_layers = len(layers_dict)
        self._set_network_layers(layers=layers_dict)

    def forward(self, x):
        start_index = self.n_layers - 1
        for i in range(start_index, -1, -1):
            sampled_layer_i = self._sample_layer(layer=getattr(self, f'layer_{i}'))
            x = sampled_layer_i(x)
            if i > 0:
                x = getattr(self, f'activation_{i}')(x)

        return x

    def _set_network_layers(self, layers: dict):
        for layer_key in layers:
            layer_dict = layers[layer_key]
            layer = nn.Linear(layer_dict['n_input'], layer_dict['n_output'], bias=True)
            layer.bias_mean = layer_dict['bias_mean']
            layer.bias_log_var = layer_dict['bias_log_var']
            layer.weight_mean = layer_dict['weight_mean']
            layer.weight_log_var = layer_dict['weight_log_var']
            setattr(self, f'layer_{layer_key}', layer)
            setattr(self, f'activation_{layer_key}', self.activation)

    @staticmethod
    def _sample_layer(layer):

        bias_dist = Normal(loc=layer.bias_mean, scale=layer.bias_log_var)
        bias_sampled = bias_dist.sample()

        weight_dist = Normal(loc=layer.weight_mean, scale=layer.weight_log_var)
        weight_sampled = weight_dist.sample()

        state_dict = layer.state_dict()
        state_dict['bias'] = bias_sampled
        state_dict['weight'] = weight_sampled
        layer.load_state_dict(state_dict)
        return layer


def _transform_genome_to_layers(nodes: dict, connections: dict, n_output: int) -> dict:
    '''
    Layers are assigned an integer key, starting with the output layer (0) towards the hidden layers (1, 2, ...)
    '''
    layers = dict()

    output_node_keys = list(nodes.keys())[:n_output]

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


def _is_next_layer_input(layer_node_keys):
    '''
    Given all the keys in a layer will return True if all the keys are negative.
    '''
    is_negative = True
    for key in layer_node_keys:
        is_negative *= True if key < 0 else False
    return is_negative


def _filter_nodes_without_input_connection(node_keys, connections: dict):
    node_keys_filtered = []
    output_node_keys = list(zip(*list(connections.keys())))[1]
    for node_key in node_keys:
        if node_key in output_node_keys:
            node_keys_filtered.append(node_key)
    return list(set(node_keys_filtered))


def build_layer_parameters(nodes, connections, layer_node_keys):
    '''
    It only works for feed-forward neural networks without multihop connections.
    That is, there is no recurrent connection neigther connections between layer{i} and layer{i+2}
    '''
    # TODO: filter out layer_node_keys that don't have any input connection
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
