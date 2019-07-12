import torch
from torch import nn
from torch.autograd import Variable

from neat.genome import GenomeSample


def _get_hidden_nodes(nodes: dict, n_output: int):
    hidden_nodes = list(nodes.keys())[n_output:]
    return hidden_nodes


def get_nn_from_genome(genome: GenomeSample) -> nn.Module:
    '''
    Given a Genome, we get the Pytorch Neural Network model
    '''
    n_output = genome.n_output
    n_input = genome.n_input
    nodes = genome.node_genes
    connections = genome.connection_genes

    hidden_node_keys = _get_hidden_nodes(nodes, n_output)

    layers = []
    if hidden_node_keys:
        pass
    else:
        layers.append(nn.Linear(n_input, n_output, bias=True))
        layers.append(nn.Sigmoid())

    network = Network(layers=layers)
    return network


class Network(nn.Module):

    def __init__(self, genome: GenomeSample):
        super(Network, self).__init__()
        self.n_output = genome.n_output
        self.n_input = genome.n_input
        self.nodes = genome.node_genes
        self.connections = genome.connection_genes

        hidden_node_keys = _get_hidden_nodes(nodes=self.nodes, n_output=self.n_output)

        layers = self._get_network_layers(n_input=self.n_input,
                                          n_output=self.n_output,
                                          nodes=self.nodes,
                                          connections=self.connections)

        self.layers = layers
        for i, layer in enumerate(layers):
            setattr(self, f'layer_{i}', layer)

        self._set_network_weights(nodes=self.nodes, connections=self.connections)

    def forward(self, x):
        # TODO: do not use
        for i, layer in enumerate(self.layers):
            x = getattr(self, f'layer_{i}')(x)

        return x

    def _set_network_weights(self, nodes, connections):

        # logger.debug('Setting Network weights')
        print('Setting Network weights')
        # https://discuss.pytorch.org/t/over-writing-weights-of-a-pre-trained-network-like-alexnet/11912
        state_dict = self.state_dict()
        state_dict["layer_0.bias"] = torch.tensor([nodes[0].bias, nodes[1].bias])
        state_dict["layer_0.weight"] = torch.tensor([[connections[(-1, 0)], connections[(-2, 0)]],
                                                     [connections[(-1, 1)], connections[(-2, 1)]]])
        self.load_state_dict(state_dict)

    def _get_network_layers(self, n_input, n_output, nodes: dict, connections: dict):
        layers = dict()

        output_node_keys = list(nodes.keys())[:n_output]
        output_layer = _get_layer_definition(nodes=nodes,
                                             connections=connections,
                                             layer_node_keys=output_node_keys)
        layers['output'] = output_layer
        # recursively get layers
        hidden_nodes = len(nodes.keys()) - n_output
        while hidden_nodes > 0:
            pass

        if self._has_hidden_layers(nodes=nodes):
            pass
        else:
            layers.append(nn.Linear(n_input, n_output, bias=True))
            layers.append(nn.Sigmoid())
        return layers

    def _has_hidden_layers(self, nodes):
        if len(nodes) > self.n_output:
            return True
        return False

def _get_layer_definition(nodes, connections, layer_node_keys):
    '''
    It only works for feed-forward neural networks without multihop connections.
    That is, there is no recurrent connection neigther connections between layer{i} and layer{i+2}
    '''
    layer = dict()
    n_output = len(layer_node_keys)

    # get bias
    layer_node_keys.sort()
    bias_values = [nodes[key].bias for key in layer_node_keys]
    bias = torch.tensor(bias_values)

    layer_connections = dict()
    input_node_keys = set({})
    for key in connections:
        output_node = key[1]
        if output_node in layer_node_keys:
            input_node_keys = input_node_keys.union({key[0]})
            layer_connections[key] = connections[key]

    n_input = len(input_node_keys)

    key_index_mapping_input = dict()
    for input_index, input_key in enumerate(input_node_keys):
        key_index_mapping_input[input_key] = input_index

    key_index_mapping_output = dict()
    for output_index, output_key in enumerate(layer_node_keys):
        key_index_mapping_output[output_key] = output_index

    weights = torch.zeros([n_input, n_output])
    for key in layer_connections:
        weights[key_index_mapping_input[key[0]], key_index_mapping_output[key[1]]] = \
            layer_connections[key]
    layer['n_input'] = n_input
    layer['n_output'] = n_output
    layer['input_keys'] = input_node_keys
    layer['output_keys'] = layer_node_keys
    layer['bias'] = bias
    layer['weights'] = weights
    return layer
