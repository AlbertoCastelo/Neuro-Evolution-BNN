from neat.genome import Genome
from neat.representation_mapping.genome_to_network.complex_stochastic_network import ComplexStochasticNetwork, \
    DEFAULT_LOGVAR


def convert_stochastic_network_to_genome(network: ComplexStochasticNetwork, original_genome: Genome = None, fitness=None,
                                         fix_std=True) -> Genome:
    if original_genome is None:
        raise ValueError('Not implemented')

    genome = original_genome.copy()

    for layer_index in range(network.n_layers):
        stochastic_linear_layer = getattr(network, f'layer_{layer_index}')
        layer = network.layers[layer_index]

        for bias_index, node_key in enumerate(layer.output_keys):
            # if node_key in genome.node_genes.keys():
            genome.node_genes[node_key].set_mean(stochastic_linear_layer.qb_mean[bias_index])
            if fix_std:
                bias_logvar = DEFAULT_LOGVAR
            else:
                bias_logvar = stochastic_linear_layer.qb_logvar[bias_index]
            genome.node_genes[node_key].set_log_var(bias_logvar)

        for connection_input_index, input_node_key in enumerate(layer.input_keys):
            for connection_output_index, output_node_key in enumerate(layer.output_keys):
                connection_key = (input_node_key, output_node_key)
                mean = stochastic_linear_layer.qw_mean[connection_output_index, connection_input_index]
                if fix_std:
                    weight_logvar = DEFAULT_LOGVAR
                else:
                    weight_logvar = stochastic_linear_layer.qw_logvar[connection_output_index, connection_input_index]
                if connection_key in genome.connection_genes.keys():
                    genome.connection_genes[connection_key].set_mean(mean)
                    genome.connection_genes[connection_key].set_log_var(weight_logvar)

    genome.fitness = fitness

    return genome

