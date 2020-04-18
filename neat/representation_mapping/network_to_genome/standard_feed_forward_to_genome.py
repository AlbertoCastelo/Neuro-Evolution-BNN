from torch import nn

from neat.genome import Genome


def get_genome_from_standard_network(network: nn.Module, key=1, std=0.00001) -> Genome:
    parameters = network.state_dict()
    genome = Genome(key=key)
    # genome._initialize_output_nodes()

    n_layers = len(parameters) // 2
    nodes_per_layer = {}
    # process nodes
    node_key_index = 0
    for i in range(n_layers):
        biases = parameters[f'layer_{i}.bias'].numpy()
        nodes_per_layer[i] = node_key_index
        for bias in biases:
            genome.add_node(key=node_key_index, mean=bias, std=std)
            node_key_index += 1
    nodes_per_layer[i+1] = -genome.n_input
    # process connections
    for i in range(n_layers):
        weights = parameters[f'layer_{i}.weight'].numpy()
        n_output, n_input = weights.shape
        output_start_index = nodes_per_layer[i]
        # if nodes_per_layer.get(i+1):
        input_start_index = nodes_per_layer[i+1]

        for output_index in range(n_output):

            for input_index in range(n_input):
                key = (input_start_index + input_index,
                       output_start_index + output_index)
                genome.add_connection(key=key,
                                      mean=weights[output_index][input_index],
                                      std=std)
    return genome

