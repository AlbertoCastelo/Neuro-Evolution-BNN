import graphviz
import matplotlib.pyplot as plt
import seaborn as sns
from neat.genome import Genome


def plot_genome_network(genome: Genome, filename='./network.png', view=True):
    return plot_network(nodes=list(genome.node_genes.keys()),
                        edges=list(genome.connection_genes.keys()),
                        input_nodes=genome.get_input_nodes_keys(),
                        output_nodes=genome.get_output_nodes_keys(),
                        filename=filename,
                        view=view)


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
    return graph


def plot_number_of_parameters(data):
    sns.boxplot(data=data, y='n_parameters', x='correlation_id')
    plt.show()