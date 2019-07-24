import torch
from torch import nn

from neat.configuration import get_configuration
from neat.genome import GenomeSample, Genome
from neat.representation.utils import get_activation


class DeterministicNetwork(nn.Module):

    def __init__(self, genome: GenomeSample):
        super(DeterministicNetwork, self).__init__()

        self.n_output = genome.n_output
        self.n_input = genome.n_input
        self.nodes = genome.node_genes
        self.connections = genome.connection_genes
        self.activation = get_activation()
        layers_dict = self._transform_genome_to_layers(nodes=self.nodes,
                                                       connections=self.connections,
                                                       n_output=self.n_output)
        self.n_layers = len(layers_dict)
        self._set_network_layers(layers=layers_dict)

        self._set_network_weights(layers=layers_dict)

    def forward(self, x):
        start_index = self.n_layers - 1
        for i in range(start_index, -1, -1):
            x = getattr(self, f'layer_{i}')(x)
            x = getattr(self, f'activation_{i}')(x)
        return x

    def _set_network_layers(self, layers: dict):
        # layers_list = list(layers.keys())
        for layer_key in layers:
            layer = layers[layer_key]
            setattr(self, f'layer_{layer_key}', nn.Linear(layer['n_input'], layer['n_output'], bias=True))
            setattr(self, f'activation_{layer_key}', self.activation)

    def _set_network_weights(self, layers: dict):
        # logger.debug('Setting Network weights')
        print('Setting Network weights')
        # https://discuss.pytorch.org/t/over-writing-weights-of-a-pre-trained-network-like-alexnet/11912
        state_dict = self.state_dict()
        for layer_key in layers:
            layer = layers[layer_key]

            state_dict[f'layer_{layer_key}.bias'] = layer['bias']
            state_dict[f'layer_{layer_key}.weight'] = layer['weights']

        self.load_state_dict(state_dict)

    def _transform_genome_to_layers(self, n_output, nodes: dict, connections: dict) -> dict:
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
        bias_values = [nodes[key] for key in layer_node_keys]
        bias = torch.tensor(bias_values)

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

        weights = torch.zeros([n_output, n_input])
        for key in layer_connections:
            weights[key_index_mapping_output[key[1]], key_index_mapping_input[key[0]]] = \
                layer_connections[key]

        # weights = torch.transpose(weights)
        layer['n_input'] = n_input
        layer['n_output'] = n_output
        layer['input_keys'] = input_node_keys
        layer['output_keys'] = layer_node_keys
        layer['bias'] = bias
        layer['weights'] = weights
        return layer

    def _has_hidden_layers(self, nodes):
        if len(nodes) > self.n_output:
            return True
        return False
