import math

import torch
from torch import nn
from torch.distributions import Normal
from torch.nn import Parameter
import torch.nn.functional as F

from neat.genome import Genome
from neat.representation.utils import get_activation


class StochasticNetwork(nn.Module):
    def __init__(self, genome: Genome):
        super(StochasticNetwork, self).__init__()
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
            x = getattr(self, f'layer_{i}')(x)
            if i > 0:
                x = getattr(self, f'activation_{i}')(x)
        return x

    def _set_network_layers(self, layers: dict):
        # layers_list = list(layers.keys())
        for layer_key in layers:
            layer_dict = layers[layer_key]
            parameters = {'qw_mean': layer_dict['weight_mean'],
                          'qw_logvar': layer_dict['weight_std'],
                          'qb_mean': layer_dict['bias_mean'],
                          'qb_logvar': layer_dict['bias_std']}
            layer = StochasticLinear(in_features=layer_dict['n_input'],
                                     out_features=layer_dict['n_output'],
                                     parameters=parameters)

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
            layer.bias_std = layer_dict['bias_std']
            layer.weight_mean = layer_dict['weight_mean']
            layer.weight_std = layer_dict['weight_std']
            setattr(self, f'layer_{layer_key}', layer)
            setattr(self, f'activation_{layer_key}', self.activation)

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
        layer = _get_layer_definition(nodes=nodes,
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


class StochasticLinear(nn.Module):

    def __init__(self, in_features, out_features, parameters=None, q_logvar_init=-5):
        # p_logvar_init, p_pi can be either
        # (list/tuples): prior model is a mixture of Gaussians components=len(p_pi)=len(p_logvar_init)
        # float: Gussian distribution
        # q_logvar_init: float, the approximate posterior is currently always a factorized gaussian
        super(StochasticLinear, self).__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.q_logvar_init = q_logvar_init

        self.qw_mean = None
        self.qw_logvar = None
        self.qb_mean = None
        self.qb_logvar = None
        self.log_alpha = None

        if parameters is None:
            # Approximate posterior weights and biases
            self.qw_mean = Parameter(torch.Tensor(out_features, in_features))
            self.qw_logvar = Parameter(torch.Tensor(out_features, in_features))

            # optionally add bias
            self.qb_mean = Parameter(torch.Tensor(out_features))
            self.qb_logvar = Parameter(torch.Tensor(out_features))

            self.log_alpha = Parameter(torch.Tensor(1, 1))

            # initialize all paramaters
            self.reset_parameters()
        else:
            # this parameters are known
            self.qw_mean = parameters['qw_mean']
            self.qw_logvar = parameters['qw_logvar']

            self.qb_mean = parameters['qb_mean']
            self.qb_logvar = parameters['qb_logvar']

            if 'log_alpha' in parameters:
                self.log_alpha = parameters['log_alpha']

    def reset_parameters(self):
        # initialize (trainable) approximate posterior parameters
        stdv = 10. / math.sqrt(self.in_features)
        self.qw_mean.data.uniform_(-stdv, stdv)
        self.qw_logvar.data.uniform_(-stdv, stdv).add_(self.q_logvar_init)
        self.qb_mean.data.uniform_(-stdv, stdv)
        self.qb_logvar.data.uniform_(-stdv, stdv).add_(self.q_logvar_init)

        # assumes 1 sigma for all weights per layer.
        self.log_alpha.data.uniform_(-stdv, stdv)

    def forward(self, x):
        # weights = self.qw_mean + torch.exp(1.0 + self.qw_logvar)
        # bias = self.qb_mean + torch.exp(1.0 + self.qb_logvar)

        x_mu_w = F.linear(input=x, weight=self.qw_mean)
        x_log_var_w = F.linear(input=x, weight=torch.exp(1.0 + self.qw_logvar))

        batch_size = x.shape[0]

        mu_b = self.qb_mean.repeat(batch_size, 1)
        log_var_b = torch.exp(1.0 + self.qb_logvar).repeat(batch_size, 1)

        output_size = x_mu_w.size()
        output = x_mu_w + x_log_var_w * torch.randn(output_size) + \
                 mu_b + log_var_b * torch.randn(output_size)

        # # combining mean for bias and weights
        # fc_q_mean = F.linear(input=input, weight=self.qw_mean, bias=self.qb_mean)
        #
        # # log std for weighs
        # fc_qw_si = torch.sqrt(1e-8 + F.linear(input=input.pow(2),
        #                                       weight=torch.exp(self.log_alpha) * self.qw_mean.pow(2)))
        #
        # # log std for bias
        # fc_qb_si = torch.sqrt(1e-8 + F.linear(input=input.pow(2),
        #                                       weight=torch.exp(self.log_alpha) * self.qb_mean.pow(2)))
        #
        # output = fc_q_mean + fc_qw_si * (torch.randn(fc_q_mean.size())) + fc_qb_si * (torch.randn(fc_q_mean.size()))
        return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'
