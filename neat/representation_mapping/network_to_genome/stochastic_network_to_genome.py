from neat.genome import Genome
from neat.representation_mapping.genome_to_network.complex_stochastic_network import ComplexStochasticNetwork


def convert_stochastic_network_to_genome(network: ComplexStochasticNetwork, original_genome: Genome = None) -> Genome:
    if original_genome is None:
        raise ValueError('Not implemented')

    genome = original_genome.copy()

    layers = network.layers
    for layer_index, layer in layers.items():
        for bias_index, node_key in enumerate(layer.output_keys):
            genome.node_genes[node_key].set_mean(layer.bias_mean[bias_index])
            genome.node_genes[node_key].set_log_var(layer.bias_log_var[bias_index])

        for connection_input_index, input_node_key in enumerate(layer.input_keys):
            for connection_output_index, output_node_key in enumerate(layer.output_keys):
                connection_key = (input_node_key, output_node_key)
                mean = layer.weight_mean[connection_output_index, connection_input_index]
                log_var = layer.weight_log_var[connection_output_index, connection_input_index]
                if connection_key in genome.connection_genes.keys():
                    genome.connection_genes[connection_key].set_mean(mean)
                    genome.connection_genes[connection_key].set_log_var(log_var)
                else:
                    assert mean == 0.0
                    assert log_var <= -10.0

    return genome

