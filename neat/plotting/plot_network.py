import graphviz


def plot_network(nodes, edges, input_nodes, output_nodes, view=False,
                 filename='./network.png', fmt='svg',
                 node_names=None, node_colors=None):
    '''
    Most code taken from: https://github.com/CodeReclaimers/neat-python/blob/master/examples/xor/visualize.py
    '''

    if node_names is None:
        node_names = {}

    if node_colors is None:
        node_colors = {}

    node_attrs = {
        'shape': 'circle',
        'fontsize': '9',
        'height': '0.2',
        'width': '0.2'}

    graph = graphviz.Digraph(format=fmt, node_attr=node_attrs)

    for node in nodes:
        name = str(node)
        if node in input_nodes:
            node_attrs = {'style': 'filled', 'shape': 'box', 'fillcolor': node_colors.get(node, 'lightgray')}
        elif node in output_nodes:
            node_attrs = {'style': 'filled', 'fillcolor': node_colors.get(node, 'lightblue')}
        else:
            node_attrs = {'style': 'filled', 'fillcolor': node_colors.get(node, 'white')}
        graph.node(name, _attributes=node_attrs)

    for edge in edges:
        input, output = edge
        a = str(input)
        b = str(output)
        color = 'green'
        style = 'solid'
        width = str(1)
        # style = 'solid' if cg.enabled else 'dotted'
        # color = 'green' if cg.weight > 0 else 'red'
        # width = str(0.1 + abs(cg.weight / 5.0))
        graph.edge(a, b, _attributes={'style': style, 'color': color, 'penwidth': width})
    graph.render(filename, view=view)
