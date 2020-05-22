import copy
import numba
from experiments.logger import logger
from neat.utils import timeit


def calculate_nodes_per_layer(links: list or tuple, output_node_keys: list, input_node_keys: list):
    '''
    You need to make sure that nodes are connected. Otherwise they won't be included.
    '''
    nodes_per_layer = {}
    nodes_per_layer[0] = output_node_keys

    layer_counter = 1
    layer_keys = output_node_keys
    is_not_done = True
    while is_not_done:
        previous_layer_keys = []
        for node_key in layer_keys:
            for in_node_key, out_node_key in links:
                if node_key == out_node_key:
                    previous_layer_keys.append(in_node_key)
        nodes_per_layer[layer_counter] = previous_layer_keys

        if _is_next_layer_input(previous_layer_keys):
            is_not_done = False
            nodes_per_layer[layer_counter] = input_node_keys
        else:
            layer_counter += 1
            layer_keys = previous_layer_keys

    layers_indices = list(nodes_per_layer.keys())
    layers_indices.sort(reverse=True)
    node_keys = set(input_node_keys)
    for layer_index in layers_indices[1:]:
        logger.debug(f'Layer index: {layer_index}')
        repeated_nodes = set(nodes_per_layer[layer_index]).intersection(node_keys)
        node_keys = node_keys.union(set(nodes_per_layer[layer_index]))
        # if len(repeated_nodes) > 0:
        logger.debug(f'Repeated_nodes: {layer_index}')
        # remove repeated_nodes from layer
        nodes_per_layer[layer_index] = list(set(nodes_per_layer[layer_index]) - repeated_nodes)

    return nodes_per_layer


def get_nodes_per_depth_level(links: list or tuple):
    max_graph_depth_per_node = calculate_max_graph_depth_per_node(links=links)

    # get nodes at each depth level
    nodes_per_depth_level = {}
    for node_key, depth in max_graph_depth_per_node.items():
        if depth in nodes_per_depth_level:
            nodes_per_depth_level[depth].append(node_key)
        else:
            nodes_per_depth_level[depth] = [node_key]

    return nodes_per_depth_level


def calculate_max_graph_depth_per_node(links: list):
    input_nodes = list(zip(*links))[0]
    output_nodes = list(zip(*links))[1]
    all_nodes = list(set(input_nodes).union(set(output_nodes)))

    connections_per_node = _get_connections_per_node(connections=links, inverse_order=True)

    # get depths_per_node
    max_graph_depth_per_node = {}
    for node_key in all_nodes:
        depth = _get_depth_per_node(node_key=node_key, connections_per_node=connections_per_node,
                                    max_graph_depth_per_node=max_graph_depth_per_node)
        max_graph_depth_per_node[node_key] = depth

    return max_graph_depth_per_node


def _get_depth_per_node(node_key: int, connections_per_node: dict, max_graph_depth_per_node: dict):
    if node_key not in connections_per_node:
        return 0

    # all parent nodes are negative
    if _is_next_layer_input(layer_node_keys=connections_per_node[node_key]):
        return 1

    max_depth = 0
    for parent_node_key in connections_per_node[node_key]:
        if parent_node_key in max_graph_depth_per_node:
            max_depth_candidate = max_graph_depth_per_node[parent_node_key] + 1
        elif parent_node_key < 0:
            max_depth_candidate = 0
        else:
            max_depth_candidate = _get_depth_per_node(node_key=parent_node_key,
                                                      connections_per_node=connections_per_node,
                                                      max_graph_depth_per_node=max_graph_depth_per_node) + 1

        if max_depth_candidate > max_depth:
            max_depth = max_depth_candidate
    return max_depth


def _is_next_layer_input(layer_node_keys):
    '''
    Given all the keys in a layer will return True if all the keys are negative.
    '''
    logger.debug(layer_node_keys)
    is_negative = True
    for key in layer_node_keys:
        is_negative *= True if key < 0 else False
    return is_negative


@timeit
def exist_cycle(connections: list) -> bool:
    # change data structure
    con = _get_connections_per_node(connections)

    def _go_throgh_graph(node_in, graph, past=[]):
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


@numba.jit
def _go_through_graph(node_in, graph, past=[]):
    if node_in in graph.keys():
        for node in graph[node_in]:
            if node in past:
                return True
            else:
                past_copy = copy.deepcopy(past)
                past_copy.append(node_in)
                if _go_through_graph(node_in=node, graph=graph, past=past_copy):
                    return True
    else:
        return False


@timeit
def exist_cycle_numba(connections: list) -> bool:
    # change data structure
    con = _get_connections_per_node(connections)
    #     print(con)
    for node_in, nodes_out in con.items():
        if _go_through_graph(node_in, graph=con, past=[]):
            return True
    return False


@timeit
def _get_connections_per_node(connections: list, inverse_order=False):
    '''
    :param connections: eg. ((-1, 1), (1, 2), (2, 3), (2, 4))
    :param inverse_order: whether it follows the input to output direction or the output to input direction
    :return: {-1: [1], 1: [2], 2: [3, 4]
    '''
    con = {}
    for connection in connections:
        input_node_key, output_node_key = connection
        if inverse_order:
            output_node_key, input_node_key = connection
        if input_node_key in con:
            con[input_node_key].append(output_node_key)
        else:
            con[input_node_key] = [output_node_key]
    return con


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
