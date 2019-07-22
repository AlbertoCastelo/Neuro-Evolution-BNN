import torch
from torch import nn
from torch.distributions import Normal

from neat.genome import Genome


class StochasticNetwork(nn.Module):
    def __init__(self, genome: Genome):
        super(StochasticNetwork, self).__init__()
        self.n_output = genome.n_output
        self.n_input = genome.n_input
        self.nodes = genome.node_genes
        self.connections = genome.connection_genes

        layers_dict = self._transform_genome_to_layers(nodes=self.nodes,
                                                       connections=self.connections,
                                                       n_output=self.n_output)
        self.n_layers = len(layers_dict)
        self._set_network_layers(layers=layers_dict)

    def forward(self, x):
        start_index = self.n_layers - 1
        for i in range(start_index, -1, -1):
            sampled_layer_i = self._sample_layer(layer=getattr(self, f'layer_{i}'))

            x = sampled_layer_i(x)
            x = getattr(self, f'activation_{i}')(x)
        return x

    @staticmethod
    def _sample_layer(layer):

        bias_dist = Normal(loc=layer.bias_mean, scale=layer.bias_std)
        bias_sampled = bias_dist.sample()

        weight_dist = Normal(loc=layer.weight_mean, scale=layer.weight_std)
        weight_sampled = weight_dist.sample()

        state_dict = layer.state_dict()
        state_dict['bias'] = bias_sampled
        state_dict['weight'] = weight_sampled
        layer.load_state_dict(state_dict)
        return layer

    def _transform_genome_to_layers(self, nodes: dict, connections: dict, n_output: int) -> dict:
        '''
        Layers are assigned an integer key, starting with the output layer (0) towards the hidden layers (1, 2, ...)
        '''
        layers = dict()

        output_node_keys = list(nodes.keys())[:n_output]

        layer_node_keys = output_node_keys

        layer_counter = 0
        is_not_finished = True
        while is_not_finished:
            layer = self._get_layer_definition(nodes=nodes,
                                               connections=connections,
                                               layer_node_keys=layer_node_keys)
            layer_node_keys = layer['input_keys']
            layers[layer_counter] = layer
            layer_counter += 1
            if self._is_next_layer_input(layer_node_keys):
                is_not_finished = False
        return layers

    def _set_network_layers(self, layers: dict):
        # layers_list = list(layers.keys())
        for layer_key in layers:
            layer_dict = layers[layer_key]
            layer = nn.Linear(layer_dict['n_input'], layer_dict['n_output'], bias=True)
            layer.bias_mean = layer_dict['bias_mean']
            layer.bias_std = layer_dict['bias_std']
            layer.weight_mean = layer_dict['weight_mean']
            layer.weight_std = layer_dict['weight_std']
            setattr(self, f'layer_{layer_key}', layer)
            setattr(self, f'activation_{layer_key}', nn.Sigmoid())

    @staticmethod
    def _is_next_layer_input(layer_node_keys):
        '''
        Given all the keys in a layer will return True if all the keys are negative.
        '''
        is_negative = True
        for key in layer_node_keys:
            is_negative *= True if key < 0 else False
        return is_negative

    @staticmethod
    def _get_layer_definition(nodes, connections, layer_node_keys):
        '''
        It only works for feed-forward neural networks without multihop connections.
        That is, there is no recurrent connection neigther connections between layer{i} and layer{i+2}
        '''
        layer = dict()
        n_output = len(layer_node_keys)

        # get bias
        layer_node_keys.sort()
        bias_mean_values = [nodes[key].bias_mean for key in layer_node_keys]
        bias_mean = torch.tensor(bias_mean_values)

        bias_std_values = [nodes[key].bias_std for key in layer_node_keys]
        bias_std = torch.tensor(bias_std_values)

        layer_connections = dict()
        input_node_keys = set({})
        for key in connections:
            output_node = key[1]
            if output_node in layer_node_keys:
                input_node_keys = input_node_keys.union({key[0]})
                layer_connections[key] = connections[key]

        input_node_keys = list(input_node_keys)
        n_input = len(input_node_keys)

        key_index_mapping_input = dict()
        for input_index, input_key in enumerate(input_node_keys):
            key_index_mapping_input[input_key] = input_index

        key_index_mapping_output = dict()
        for output_index, output_key in enumerate(layer_node_keys):
            key_index_mapping_output[output_key] = output_index

        weight_mean = torch.zeros([n_output, n_input])
        weight_std = torch.zeros([n_output, n_input])
        for key in layer_connections:
            weight_mean[key_index_mapping_output[key[1]], key_index_mapping_input[key[0]]] = \
                layer_connections[key].weight_mean
            weight_std[key_index_mapping_output[key[1]], key_index_mapping_input[key[0]]] = \
                layer_connections[key].weight_std

        # weights = torch.transpose(weights)
        layer['n_input'] = n_input
        layer['n_output'] = n_output
        layer['input_keys'] = input_node_keys
        layer['output_keys'] = layer_node_keys
        layer['bias_mean'] = bias_mean
        layer['bias_std'] = bias_std
        layer['weight_mean'] = weight_mean
        layer['weight_std'] = weight_std
        return layer
