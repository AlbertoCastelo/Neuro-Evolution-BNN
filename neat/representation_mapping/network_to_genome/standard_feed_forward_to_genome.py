from torch import nn

from neat.genome import Genome



def get_genome_from_standard_network(network: nn.Module, key=1, std=0.00001) -> Genome:
    parameters = network.state_dict()
    genome = Genome(key=key)

    n_layers = len(parameters) // 2
    nodes_per_layer = {}
    # process nodes
    node_key_index = 0
    for i in range(n_layers):
        biases = parameters[f'layer_{i}.bias'].numpy()
        nodes_index = list(range(node_key_index, node_key_index + len(biases)))
        nodes_per_layer[i] = nodes_index
        node_key_index += len(nodes_index)
        for bias, key in zip(biases, nodes_index):
            genome.add_node(key=key, mean=bias, std=std)
            # node_key_index += 1
    # nodes_per_layer[i+1] = list(range(-1, -genome.n_input - 1, -1))
    nodes_per_layer[i+1] = list(range(-genome.n_input, 0))

    # process connections
    for i in range(n_layers):
        weights = parameters[f'layer_{i}.weight'].numpy()
        n_output, n_input = weights.shape
        output_keys = nodes_per_layer[i]
        # if nodes_per_layer.get(i+1):
        input_keys = nodes_per_layer[i+1]

        for output_key, output_index in zip(output_keys, range(n_output)):

            for input_key, input_index in zip(input_keys, range(n_input)):
                key = (input_key, output_key)
                genome.add_connection(key=key,
                                      mean=weights[output_index][input_index],
                                      std=std)
    return genome
