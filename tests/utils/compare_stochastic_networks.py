from neat.representation_mapping.genome_to_network.complex_stochastic_network import ComplexStochasticNetwork


def compare_networks(network_1: ComplexStochasticNetwork, network_2: ComplexStochasticNetwork):
    if network_1.n_layers != network_2.n_layers:
        print('Different layers')
        return False

    for i in range(network_1.n_layers):
        print(f'LAYER-{i}')
        layer_self = getattr(network_1, f'layer_{i}')
        layer_other = getattr(network_2, f'layer_{i}')

        print('W-MEANs')
        print('Network 1')
        print(layer_self.qw_mean)
        print('Network 2')
        print(layer_other.qw_mean)

        print('W-LOGVARs')
        print('Network 1')
        print(layer_self.qw_logvar)
        print('Network 2')
        print(layer_other.qw_logvar)

        print('B-MEANs')
        print('Network 1')
        print(layer_self.qb_mean)
        print('Network 2')
        print(layer_other.qb_mean)

        print('B-LOGVARs')
        print('Network 1')
        print(layer_self.qb_logvar)
        print('Network 2')
        print(layer_other.qb_logvar)