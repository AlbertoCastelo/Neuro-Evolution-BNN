import torch
from torch import nn
from experiments.logger import logger
from neat.genome import Genome
from neat.representation_mapping.genome_to_network.graph_utils import calculate_nodes_per_layer
from neat.representation_mapping.genome_to_network.layers import StochasticLinearParameters, \
    ComplexStochasticLinear, StochasticLinearMasks
from neat.representation_mapping.genome_to_network.utils import get_activation
from neat.utils import timeit

DEFAULT_LOGVAR = -33.0


class ComplexStochasticNetwork(nn.Module):
    def __init__(self, genome: Genome, is_trainable=False, is_cuda=False):
        super(ComplexStochasticNetwork, self).__init__()
        self.is_trainable = is_trainable
        self.is_cuda = is_cuda
        self.genome = genome
        self.n_output = genome.n_output
        self.n_input = genome.n_input
        self.nodes = genome.node_genes
        self.connections = genome.connection_genes
        self.config = genome.genome_config
        self.activation = get_activation(activation=self.config.node_activation)
        self.layers = transform_genome_to_layers(genome=genome)
        self.layers_masks = generate_layer_masks(self.layers, genome, fix_std=self.config.fix_std)
        self.n_layers = len(self.layers)
        self._set_network_layers(layers=self.layers, layers_masks=self.layers_masks)
        self._cache = {}

    def forward(self, x):
        kl_qw_pw = 0.0
        start_index = self.n_layers - 1

        for i in range(start_index, -1, -1):
            # cache needed values from past layers
            for index_to_cache in self.layers[i].indices_of_nodes_to_cache:
                if self.is_cuda:
                    index = torch.cuda.LongTensor((index_to_cache,))
                else:
                    index = torch.LongTensor((index_to_cache,))
                self._cache[(i, index_to_cache)] = x.index_select(1, index)
            # append needed values for next layers
            chunks = [x]
            for index_needed in self.layers[i].indices_of_needed_nodes:
                chunks.append(self._cache[index_needed])
            x = torch.cat(chunks, 1)
            x, kl_qw_pw_layer = getattr(self, f'layer_{i}')(x)
            kl_qw_pw += kl_qw_pw_layer
            if i > 0:
                x = getattr(self, f'activation_{i}')(x)
        return x, kl_qw_pw

    def _set_network_layers(self, layers: dict, layers_masks: dict):
        for layer_key in layers:
            layer = layers[layer_key]
            layer_masks = layers_masks[layer_key]
            parameters = StochasticLinearParameters.create(qw_mean=layer.weight_mean,
                                                           qw_logvar=layer.weight_log_var,
                                                           qb_mean=layer.bias_mean,
                                                           qb_logvar=layer.bias_log_var)

            layer = ComplexStochasticLinear(in_features=layer.n_input,
                                            out_features=layer.n_output,
                                            parameters=parameters,
                                            masks=layer_masks,
                                            is_cuda=self.is_cuda,
                                            is_trainable=self.is_trainable)

            setattr(self, f'layer_{layer_key}', layer)
            setattr(self, f'activation_{layer_key}', self.activation)

    def clear_non_existing_weights(self, clear_grad=True):
        # clear gradients and values of non-existing connections on the genome
        # THIS HAS A PROBLEM: IN-PLACE OPERATIONS DON'T WORK WELL WITH AUTOGRAD
        for layer_index in range(self.n_layers):
            stochastic_linear_layer = getattr(self, f'layer_{layer_index}')
            layer = self.layers[layer_index]
            for connection_input_index, input_node_key in enumerate(layer.input_keys):
                for connection_output_index, output_node_key in enumerate(layer.output_keys):
                    connection_key = (input_node_key, output_node_key)
                    if connection_key not in self.genome.connection_genes.keys():
                        if clear_grad:
                            # clear gradients
                            stochastic_linear_layer.qw_mean.grad[connection_output_index, connection_input_index] = 0.0
                            stochastic_linear_layer.qw_logvar.grad[connection_output_index, connection_input_index] = 0.0

                        # # clear values
                        stochastic_linear_layer.qw_mean[connection_output_index, connection_input_index] = 0.0
                        stochastic_linear_layer.qw_logvar[connection_output_index, connection_input_index] = \
                            DEFAULT_LOGVAR


def equal(network_1, network_2, skip_logvar=True):
    if network_1.n_layers != network_2.n_layers:
        return False

    for i in range(network_1.n_layers):
        layer_self = getattr(network_1, f'layer_{i}')
        layer_other = getattr(network_2, f'layer_{i}')

        # check weights and biases are equal
        if not torch.allclose(layer_self.qw_mean, layer_other.qw_mean, atol=1e-03):
            return False
        if not torch.allclose(layer_self.qb_mean, layer_other.qb_mean, atol=1e-03):
            return False

        if not skip_logvar:
            if not torch.allclose(layer_self.qw_logvar, layer_other.qw_logvar, atol=1e-03):
                return False
            if not torch.allclose(layer_self.qb_logvar, layer_other.qb_logvar, atol=1e-03):
                return False

    return True


@timeit
def transform_genome_to_layers(genome: Genome) -> dict:
    layers = dict()
    nodes = genome.node_genes
    connections = genome.connection_genes

    nodes_per_layer = calculate_nodes_per_layer(links=list(connections.keys()),
                                                input_node_keys=genome.get_input_nodes_keys(),
                                                output_node_keys=genome.get_output_nodes_keys())
    layer_indices = list(nodes_per_layer.keys())
    layer_indices.sort()
    for layer_index in layer_indices[:-1]:
        original_nodes_in_layer = nodes_per_layer[layer_index]
        layer = LayerBuilder(nodes=nodes,
                             connections=connections,
                             layer_node_keys=original_nodes_in_layer,
                             nodes_per_layer=nodes_per_layer,
                             layer_counter=layer_index) \
            .create() \
            .get_layer()

        layers[layer_index] = layer

    # enrich layers
    for layer_counter, layer in layers.items():
        # logger.debug(f'Layer: {layer_counter}')
        # add needed indices
        for node_key in layer.external_input_keys:
            index = None
            for layer_2 in layers.values():
                if node_key in layer_2.original_input_keys:
                    index = (layer_2.key, layer_2.input_keys.index(node_key))
                    break
            assert index is not None
            layer.indices_of_needed_nodes.append(index)
            layer.needed_nodes[node_key] = index

        # add indices to cache
        for node_key in layer.original_input_keys:
            for layer_2 in layers.values():
                if node_key in layer_2.external_input_keys:
                    index = layer.input_keys.index(node_key)
                    # add if not in list
                    if index not in layer.indices_of_nodes_to_cache:
                        layer.indices_of_nodes_to_cache.append(index)

        if len(layer.indices_of_needed_nodes) > 1:
            needed_node_keys = list(layer.needed_nodes.keys())
            needed_node_keys.sort()
            sorted_indices_of_needed_nodes = []
            for node_key in needed_node_keys:
                sorted_indices_of_needed_nodes.append(layer.needed_nodes[node_key])

            assert len(sorted_indices_of_needed_nodes) == len(layer.indices_of_needed_nodes)
            layer.indices_of_needed_nodes = sorted_indices_of_needed_nodes

        logger.debug(f'Indices to cache: {layer.indices_of_nodes_to_cache}')
        logger.debug(f'Indices needed from cache: {layer.indices_of_needed_nodes}')

    return layers


def generate_layer_masks(layers: dict, genome: Genome, fix_std: bool):
    masks_per_layer = {}
    for layer_index, layer in layers.items():
        tensor_sizes = len(layer.output_keys), len(layer.input_keys)
        mask_mean = torch.ones(tensor_sizes)

        if fix_std:
            mask_logvar_mul = torch.zeros(tensor_sizes)
            mask_logvar_add = DEFAULT_LOGVAR * torch.ones(tensor_sizes)
        else:
            mask_logvar_mul = torch.ones(tensor_sizes)
            mask_logvar_add = torch.zeros(tensor_sizes)
        for connection_input_index, input_node_key in enumerate(layer.input_keys):
            for connection_output_index, output_node_key in enumerate(layer.output_keys):
                connection_key = (input_node_key, output_node_key)
                if connection_key not in genome.connection_genes.keys():

                    mask_mean[connection_output_index, connection_input_index] = 0.0
                    if not fix_std:
                        mask_logvar_add[connection_output_index, connection_input_index] = DEFAULT_LOGVAR

        masks_per_layer[layer_index] = StochasticLinearMasks(mask_mean=mask_mean, mask_logvar_mul=mask_logvar_mul,
                                                             mask_logvar_add=mask_logvar_add)
    return masks_per_layer


def _get_node_index_to_cache(node_key, layers):
    for layer in layers.values():
        if node_key in layer.original_input_keys:
        # if node_key in layer.input_keys:
            return layer.input_keys.index(node_key)


class Layer:
    def __init__(self, key, n_input, n_output):
        self.key = key
        self.n_input = n_input
        self.n_output = n_output
        self.input_keys = None
        self.external_input_keys = None
        self.original_input_keys = None
        self.output_keys = None

        self.needed_nodes = {}
        self.indices_of_nodes_to_cache = []
        self.indices_of_needed_nodes = []

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

    def add_index_to_cache(self, index):
        self.indices_of_nodes_to_cache.append(index)

    def add_index_needed(self, index):
        self.indices_of_needed_nodes.append(index)

    def validate(self):
        # validate shapes
        pass


class LayerBuilder:

    def __init__(self, nodes, connections, layer_node_keys, nodes_per_layer, layer_counter):
        '''
            :param nodes: dictionary with all the nodes in the genome
            :param connections: dictionary with all the connections of the genome
            :param layer_node_keys: a list of keys of the nodes in a layer
        '''
        self.nodes = nodes
        self.connections = connections
        self.layer_node_keys = layer_node_keys
        self.nodes_per_layer = nodes_per_layer
        self.layer_counter = layer_counter
        self._total_number_of_layers = len(nodes_per_layer) - 1

        self.layer = None

    def create(self):
        # layer_node_keys = _filter_nodes_without_input_connection(node_keys=self.layer_node_keys,
        #                                                          connections=self.connections)
        layer_node_keys = self.nodes_per_layer[self.layer_counter]

        layer_node_keys.sort()
        n_output = len(layer_node_keys)

        layer_connections = dict()
        input_node_keys = set(self.nodes_per_layer[self.layer_counter+1])
        for connection_key, connection in self.connections.items():
            input_node_key, output_node_key = connection_key
            if output_node_key in layer_node_keys:
                input_node_keys = input_node_keys.union({input_node_key})
                if connection.enabled:
                    layer_connections[connection_key] = self.connections[connection_key]

        input_node_keys = list(input_node_keys)
        n_input = len(input_node_keys)
        self.layer = Layer(key=self.layer_counter, n_input=n_input, n_output=n_output)

        # sorted input keys
        logger.debug(f'Layer: {self.layer_counter}')
        original_input_keys = self.nodes_per_layer[self.layer_counter+1]

        # TODO: why do i need this sort?
        original_input_keys.sort()
        external_input_keys = self._get_external_input_keys(input_node_keys, original_input_keys)
        input_node_keys = original_input_keys + external_input_keys
        logger.debug(f'   Input Keys: {input_node_keys}')

        self.layer.input_keys = input_node_keys
        self.layer.original_input_keys = original_input_keys
        self.layer.external_input_keys = external_input_keys
        self.layer.output_keys = layer_node_keys

        # set parameters
        self.layer.weight_mean, self.layer.weight_log_var = \
            self._build_weight_tensors(layer_connections=layer_connections,
                                       input_node_keys=input_node_keys,
                                       layer_node_keys=layer_node_keys,
                                       n_input=n_input,
                                       n_output=n_output)

        self.layer.bias_mean, self.layer.bias_log_var = self._build_bias_tensors(layer_node_keys, self.nodes)

        self.layer.validate()
        return self

    def _get_external_input_keys(self, input_node_key, original_input_keys):
        external_input_keys = list(set(input_node_key) - set(original_input_keys))
        external_input_keys.sort()
        logger.debug(f'   External Input Keys: {external_input_keys}')
        return external_input_keys

    def get_layer(self):
        return self.layer

    @staticmethod
    def _build_weight_tensors(layer_connections, input_node_keys, layer_node_keys, n_input, n_output):
        '''
        The input parameters (input_node_keys and layer_node_keys) must be sorted. Then the output matrices will
        also be sorted.

        input_node_keys = (input_node_keys_{original} | input_node_keys_{external})
        W = (W_{original} | W_{external})
        '''
        key_index_mapping_input = dict()
        for input_index, input_key in enumerate(input_node_keys):
            key_index_mapping_input[input_key] = input_index
        key_index_mapping_output = dict()
        for output_index, output_key in enumerate(layer_node_keys):
            key_index_mapping_output[output_key] = output_index
        weight_mean = torch.zeros([n_output, n_input])
        weight_log_var = DEFAULT_LOGVAR * torch.ones([n_output, n_input])
        for connection_key in layer_connections:
            input_node_key, output_node_key = connection_key
            weight_mean[key_index_mapping_output[output_node_key], key_index_mapping_input[input_node_key]] = \
                float(layer_connections[connection_key].get_mean())
            weight_log_var[key_index_mapping_output[output_node_key], key_index_mapping_input[input_node_key]] = \
                float(layer_connections[connection_key].get_log_var())
        return weight_mean, weight_log_var

    @staticmethod
    def _build_bias_tensors(layer_node_keys, nodes):
        '''
        This also keeps the order
        '''
        bias_mean_values = [nodes[key].get_mean() for key in layer_node_keys]
        bias_mean = torch.tensor(bias_mean_values)

        bias_log_var_values = [nodes[key].get_log_var() for key in layer_node_keys]
        bias_log_var = torch.tensor(bias_log_var_values)
        return bias_mean, bias_log_var


def _generate_cache_index(layer_counter: int, index: int):
    return (layer_counter, index)


def _get_layer_given_node(nodes_per_layer, node_key):
    for layer, node_keys in nodes_per_layer.items():
        if node_key in node_keys:
            return layer
