from neat.genome import Genome
from neat.representation_mapping.genome_to_network.complex_stochastic_network import ComplexStochasticNetwork
from neat.representation_mapping.genome_to_network.graph_utils import calculate_nodes_per_layer


def convert_stochastic_network_to_genome(network: ComplexStochasticNetwork, original_genome: Genome = None) -> Genome:
    if original_genome is None:
        raise ValueError('Not implemented')

    genome = original_genome.copy()

    nodes = genome.node_genes
    connections = genome.connection_genes

    nodes_per_layer = calculate_nodes_per_layer(links=list(connections.keys()),
                                                input_node_keys=genome.get_input_nodes_keys(),
                                                output_node_keys=genome.get_output_nodes_keys())

    layers = network.layers
    for layer_index, layer in layers.items():
        for bias_index, node_key in enumerate(layer.output_keys):
            genome.node_genes[node_key].set_mean(layer.bias_mean[bias_index])
            genome.node_genes[node_key].set_log_var(layer.bias_log_var[bias_index])

        for connection_input_index, input_node_key in enumerate(layer.input_keys):
            for connection_output_index, output_node_key in enumerate(layer.output_keys):
                mean = layer.weight_mean[connection_output_index, connection_input_index]
                log_var = layer.weight_log_var[connection_output_index, connection_input_index]
                genome.connection_genes[(input_node_key, output_node_key)].set_mean(mean)
                genome.connection_genes[(input_node_key, output_node_key)].set_log_var(log_var)

    # n_layers = len(nodes_per_layer)
    # for layer, node_keys in nodes_per_layer.items():
    #     if layer == n_layers - 1:
    #         continue




    return genome

