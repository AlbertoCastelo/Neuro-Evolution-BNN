import random

from experiments.logger import logger
from neat.configuration import get_configuration
from neat.gene import ConnectionGene, Gene, NodeGene
from neat.genome import Genome
from neat.utils import timeit


class Crossover:

    def __init__(self):
        self.config = get_configuration()

    @timeit
    def get_offspring(self, offspring_key, genome1: Genome, genome2: Genome):
        assert isinstance(genome1.fitness, (int, float))
        assert isinstance(genome2.fitness, (int, float))
        if genome1.fitness > genome2.fitness:
            parent1, parent2 = genome1, genome2
        else:
            parent1, parent2 = genome2, genome1

        offspring = Genome(key=offspring_key)

        # Inherit connection genes
        for key, cg1 in parent1.connection_genes.items():
            cg2 = parent2.connection_genes.get(key)
            if cg2 is None:
                # Excess or disjoint gene: copy from the fittest parent.
                logger.debug('Use connection copy')
                offspring.connection_genes[key] = cg1.copy()
            else:
                # Homologous gene: combine genes from both parents.
                offspring.connection_genes[key] = self._get_connection_crossover(cg1, cg2)

        # Inherit node genes
        for key, ng1 in parent1.node_genes.items():
            ng2 = parent2.node_genes.get(key)
            assert key not in offspring.node_genes
            if ng2 is None:
                # Extra gene: copy from the fittest parent
                logger.debug('Use node copy')
                offspring.node_genes[key] = ng1.copy()
            else:
                # Homologous gene: combine genes from both parents.
                offspring.node_genes[key] = self._get_node_crossover(ng1, ng2)
        return offspring

    def _get_node_crossover(self, node_1: NodeGene, node_2: NodeGene):
        assert node_1.key == node_2.key
        node_key = node_1.key

        new_node = NodeGene(key=node_key)

        for attribute in new_node.crossover_attributes:
            if random.random() > 0.5:
                self.set_child_attribute(attribute=attribute, new_gene=new_node, parent_gene=node_1)
            else:
                self.set_child_attribute(attribute=attribute, new_gene=new_node, parent_gene=node_2)
                # setattr(new_node, attribute, getattr(node_2, attribute))
        return new_node

    @staticmethod
    def set_child_attribute(attribute: str, new_gene: Gene, parent_gene: Gene):
        getattr(new_gene, f'set_{attribute}')(getattr(parent_gene, f'get_{attribute}')())

    def _get_connection_crossover(self, connection_1: ConnectionGene, connection_2: ConnectionGene):
        assert connection_1.key == connection_2.key
        connection_key = connection_1.key

        new_connection = ConnectionGene(key=connection_key)
        for attribute in new_connection.crossover_attributes:
            if random.random() > 0.5:
                self.set_child_attribute(attribute=attribute, new_gene=new_connection, parent_gene=connection_1)
            else:
                self.set_child_attribute(attribute=attribute, new_gene=new_connection, parent_gene=connection_2)

        return new_connection
