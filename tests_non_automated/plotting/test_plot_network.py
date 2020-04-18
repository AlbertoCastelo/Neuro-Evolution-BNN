from neat.analysis.plotting.plot_network import plot_network


def main():
    nodes = [-1, -2, 0, 1, 2, 3]
    edges = ((-2, 1), (-1, 0), (-2, 0), (-1, 2), (2, 3), (3, 1), (3, 0))
    input_nodes = [-1, -2]
    output_nodes = [0, 1]
    plot_network(nodes, edges, input_nodes, output_nodes, view=True)


if __name__ == '__main__':
    main()