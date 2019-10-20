import copy
import itertools
import random

from experiments.logger import logger
from neat.configuration import get_configuration, BaseConfiguration
from neat.gene import NodeGene, ConnectionGene
from neat.genome import Genome


class Mutation:

    def __init__(self):
        self.config = get_configuration()

        self.fix_architecture = self.config.fix_architecture
        self.single_structural_mutation = self.config.single_structural_mutation
        self.mutate_rate = self.config.mutate_rate
        self.mutate_power = self.config.mutate_power
        self.replace_rate = self.config.replace_rate

        self.node_add_prob = self.config.node_add_prob
        self.node_delete_prob = self.config.node_delete_prob
        self.connection_add_prob = self.config.connection_add_prob
        self.connection_delete_prob = self.config.connection_delete_prob

    def mutate(self, genome: Genome):
        if not self.fix_architecture:
            genome = self._mutate_architecture(genome)

        # Mutate connection genes.
        for key in genome.connection_genes.keys():
            genome.connection_genes[key].mutate()
        # Mutate node genes (bias, response, etc.).
        for key in genome.node_genes.keys():
            genome.node_genes[key].mutate()
        return genome

    def _mutate_architecture(self, genome: Genome):
        if self.single_structural_mutation:
            div = max(1, (self.node_add_prob + self.node_delete_prob +
                          self.connection_add_prob + self.connection_delete_prob))
            r = random.random()
            if r < (self.node_add_prob / div):
                genome = self.mutate_add_node(genome)
            elif r < ((self.node_add_prob + self.node_delete_prob) / div):
                genome = self.mutate_delete_node(genome)
            elif r < ((self.node_add_prob + self.node_delete_prob +
                       self.connection_add_prob) / div):
                genome = self.mutate_add_connection(genome)
            elif r < ((self.node_add_prob + self.node_delete_prob +
                       self.connection_add_prob + self.connection_delete_prob) / div):
                genome = self.mutate_delete_connection(genome)
        else:
            if random.random() < self.node_add_prob:
                genome = self.mutate_add_node(genome)

            if random.random() < self.node_delete_prob:
                genome = self.mutate_delete_node(genome)

            if random.random() < self.connection_add_prob:
                genome = self.mutate_add_connection(genome)

            if random.random() < self.connection_delete_prob:
                genome = self.mutate_delete_connection(genome)
        return genome

    def mutate_add_node(self, genome: Genome):
        logger.debug('Mutation: Add a Node')
        # TODO: careful! this can add multihop-jumps to the network

        # Choose a random connection to split
        connection_to_split_key = random.choice(list(genome.connection_genes.keys()))

        new_node_key = genome.get_new_node_key()
        new_node = NodeGene(key=new_node_key).random_initialization()
        genome.node_genes[new_node_key] = new_node

        connection_to_split = genome.connection_genes[connection_to_split_key]
        i, o = connection_to_split_key

        # add connection between input and the new node
        # TODO: careful I'm doing a random initialization but python-neat sets weight to 1
        new_connection_key_i = (i, new_node_key)
        new_connection_i = ConnectionGene(key=new_connection_key_i)
        new_connection_i.set_mean(mean=1.0), new_connection_i.set_std(std=0.000001)
        genome.connection_genes[new_connection_key_i] = new_connection_i
        # genome.add_connection(config, i, new_node_key, 1.0, True)

        # add connection between new node and output
        new_connection_key_o = (new_node_key, o)
        new_connection_o = ConnectionGene(key=new_connection_key_o).random_initialization()
        new_connection_o.set_mean(mean=connection_to_split.get_mean())
        new_connection_o.set_std(std=connection_to_split.get_std())
        genome.connection_genes[new_connection_key_o] = new_connection_o
        # self.add_connection(config, new_node_id, o, conn_to_split.weight, True)

        # delete connection
        # Careful: neat-python disable the connection instead of deleting
        # conn_to_split.enabled = False
        del genome.connection_genes[connection_to_split_key]

        return genome

    def mutate_delete_node(self, genome: Genome):
        logger.debug('Mutation: Delete a Node')
        # Do nothing if there are no non-output nodes.
        available_nodes = self._get_available_nodes_to_be_deleted(genome)
        if not available_nodes:
            return genome

        del_key = random.choice(available_nodes)

        # TODO: if delete a node, we should
        # delete connections related
        connections_to_delete = []
        for k, v in genome.connection_genes.items():
            if del_key in v.key:
                connections_to_delete.append(v.key)

        for key in connections_to_delete:
            del genome.connection_genes[key]

        # delete node
        del genome.node_genes[del_key]
        return genome

    def mutate_add_connection(self, genome: Genome):
        logger.debug('Mutation: Add a Connection')
        possible_outputs = list(genome.node_genes.keys())
        out_node_key = random.choice(possible_outputs)

        possible_inputs = self._calculate_possible_inputs_when_adding_connection(genome,
                                                                                 out_node_key=out_node_key,
                                                                                 config=self.config)
        if len(possible_inputs) == 0:
            return genome
        in_node = random.choice(possible_inputs)

        new_connection_key = (in_node, out_node_key)

        new_connection = ConnectionGene(key=new_connection_key).random_initialization()
        genome.connection_genes[new_connection_key] = new_connection
        return genome

    def mutate_delete_connection(self, genome: Genome):
        logger.debug('Mutation: Delete a Connection')
        possible_connections_to_delete = self._calculate_possible_connections_to_delete(genome=genome)
        if len(possible_connections_to_delete) > 1:
            key = random.choice(possible_connections_to_delete)
            del genome.connection_genes[key]
        return genome

    @staticmethod
    def _calculate_possible_connections_to_delete(genome):
        '''
        Assumes the network does not have cycles nor multi-hop jumps.
        '''
        possible_connections_to_delete_set = set(genome.connection_genes.keys())
        possible_connections_to_delete_set = \
            Mutation._remove_connection_that_introduces_multihop_jumps(
                genome=genome,
                possible_connection_set=possible_connections_to_delete_set)

        possible_connections_to_delete_set = \
            Mutation._remove_connection_that_introduces_cycles(
                genome=genome,
                possible_connection_set=possible_connections_to_delete_set)

        return list(possible_connections_to_delete_set)

    def _get_available_nodes_to_be_deleted(self, genome: Genome):
        available_nodes = set(genome.node_genes.keys()) - set(genome.get_output_nodes_keys())
        return list(available_nodes)

    @staticmethod
    def _calculate_possible_inputs_when_adding_connection(genome: Genome, out_node_key: int, config: BaseConfiguration):
        # all nodes
        possible_input_keys_set = set(genome.node_genes.keys()).union(set(genome.get_input_nodes_keys()))

        # no connection between two output nodes
        possible_input_keys_set -= set(genome.get_output_nodes_keys())

        if config.feed_forward:
            # avoid self-recurrency
            possible_input_keys_set -= {out_node_key}
            # pass

        # REMOVE POSSIBLE CONNECTIONS
        possible_connection_set = set(itertools.product(list(possible_input_keys_set), [out_node_key]))

        # remove already existing connections: don't duplicate connections
        possible_connection_set -= set(genome.connection_genes.keys())

        # remove possible connections that introduce cycles
        possible_connection_set = \
            Mutation._remove_connection_that_introduces_cycles(genome=genome,
                                                               possible_connection_set=possible_connection_set)

        # remove possible connections that introduce multihop jumps
        possible_connection_set = \
            Mutation._remove_connection_that_introduces_multihop_jumps(genome=genome,
                                                                       possible_connection_set=possible_connection_set)
        if len(possible_connection_set) == 0:
            return []
        possible_input_keys_set = list(zip(*list(possible_connection_set)))[0]
        return possible_input_keys_set

    @staticmethod
    def _remove_connection_that_introduces_cycles(genome: Genome, possible_connection_set: set) -> set:
        connections_to_remove = []
        for connection in possible_connection_set:
            connections = list(genome.connection_genes.keys()) + [connection]

            if exist_cycle(connections=connections):
                connections_to_remove.append(connection)
        possible_connection_set -= set(connections_to_remove)
        return possible_connection_set

    @staticmethod
    def _remove_connection_that_introduces_multihop_jumps(genome: Genome, possible_connection_set: set) -> set:
        output_node_keys = genome.get_output_nodes_keys()
        input_node_keys = genome.get_input_nodes_keys()
        connections_to_remove = []
        for connection in possible_connection_set:
            connections = list(genome.connection_genes.keys()) + [connection]

            if adds_multihop_jump(connections=connections,
                                  output_node_keys=output_node_keys,
                                  input_node_keys=input_node_keys):
                connections_to_remove.append(connection)
        possible_connection_set -= set(connections_to_remove)
        return possible_connection_set


def adds_multihop_jump(connections: list, output_node_keys, input_node_keys) -> bool:
    layer_counter = 0
    layers = {0: output_node_keys}

    nodes = []
    layer_output_node_keys = copy.deepcopy(output_node_keys)
    is_not_done = True
    while is_not_done:

        nodes_in_previous_layer_set = set()
        for connection in connections:
            input_node = connection[0]
            output_node = connection[1]
            if output_node in layer_output_node_keys:
                if input_node in nodes:
                    return True
                nodes_in_previous_layer_set = nodes_in_previous_layer_set.union({input_node})
        nodes += list(nodes_in_previous_layer_set)

        # keep
        if len(nodes_in_previous_layer_set) == 0:
            is_not_done = False
        else:
            layer_counter += 1
            layers[layer_counter] = nodes_in_previous_layer_set
            layer_output_node_keys = nodes_in_previous_layer_set

    return False


def exist_cycle(connections: list) -> bool:
    # change data structure
    con = {}
    for connection in connections:
        if connection[0] in con:
            con[connection[0]].append(connection[1])
        else:
            con[connection[0]] = [connection[1]]

    def _go_throgh_graph(node_in, graph, past=[]):
        # print(f'{node_in}. {past}')
        if node_in in graph.keys():
            for node in graph[node_in]:
                if node in past:
                    return True
                else:
                    past_copy = copy.deepcopy(past)
                    past_copy.append(node_in)
                    if _go_throgh_graph(node_in=node, graph=graph, past=past_copy):
                        return True
        else:
            return False

    for node_in, nodes_out in con.items():
        if _go_throgh_graph(node_in, graph=con, past=[]):
            return True
    return False
