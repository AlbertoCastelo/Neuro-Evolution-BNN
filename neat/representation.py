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
        n_output = genome.n_output
        n_input = genome.n_input
        nodes = genome.node_genes
        connections = genome.connection_genes

        hidden_node_keys = _get_hidden_nodes(nodes, n_output)

        layers = self._get_network_layers(n_input=n_input,
                                          n_output=n_output,
                                          nodes=nodes,
                                          connections=connections)

        self.layers = layers
        for i, layer in enumerate(layers):
            setattr(self, f'layer_{i}', layer)

        self._set_network_weights(nodes=nodes, connections=connections)

    def _set_network_weights(self, nodes, connections):

        # logger.debug('Setting Network weights')
        print('Setting Network weights')
        # https://discuss.pytorch.org/t/over-writing-weights-of-a-pre-trained-network-like-alexnet/11912
        state_dict = self.state_dict()
        # state_dict["layer_0.bias"] = torch.tensor([nodes[0].bias, nodes[1].bias])
        state_dict["layer_0.bias"] = torch.tensor([0.0, 0.0])
        state_dict["layer_0.weight"] = torch.tensor([[connections[(-1, 0)], connections[(-2, 0)]],
                                                     [connections[(-1, 1)], connections[(-2, 1)]]])
        self.load_state_dict(state_dict)

    def _get_network_layers(self, n_input, n_output, nodes: dict, connections: dict):
        layers = []
        if self._has_hidden_layers(nodes=nodes):
            pass
        else:
            layers.append(nn.Linear(n_input, n_output, bias=True))
            layers.append(nn.Sigmoid())
        return layers

    def forward(self, x):
        # TODO: do not use
        for i, layer in enumerate(self.layers):
            x = getattr(self, f'layer_{i}')(x)

        return x

    def _has_hidden_layers(self, nodes):
        return False
