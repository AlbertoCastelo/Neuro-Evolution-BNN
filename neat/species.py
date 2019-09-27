from itertools import count
import numpy as np
from neat.configuration import get_configuration
from neat.genome import Genome


class SpeciationEngine:
    '''
    Speciation in NEAT
    https://apps.cs.utexas.edu/tech_reports/reports/tr/TR-1972.pdf
    '''

    def __init__(self):
        self.config = get_configuration()
        self.indexer = count(1)
        self.species = {}
        self.genome_to_species = {}

        self.compatibility_threshold = self.config.compatibility_threshold

        self.gdmean = None
        self.gdstdev = None

    def speciate(self, population, generation):
        """
        Disclaimer: code copied from NEAT-Python: https://neat-python.readthedocs.io/en/latest/

        Place genomes into species by genetic similarity.

        Note that this method assumes the current representatives of the species are from the old
        generation, and that after speciation has been performed, the old representatives should be
        dropped and replaced with representatives from the new generation.  If you violate this
        assumption, you should make sure other necessary parts of the code are updated to reflect
        the new behavior.
        """
        unspeciated_genomes = list(population.keys())
        distances = DistanceCalculation()
        new_representatives = {}
        new_members = {}

        # using last generation species it search for the distance between the genomes in the new population and the
        # representative from the specie
        for species_key, specie in self.species.items():
            candidates = []
            for genome_key in unspeciated_genomes:
                genome = population[genome_key]
                d = distances.get_distance(genome_0=specie.representative, genome_1=genome)
                candidates.append((d, genome))

            # The new representative is the genome closest to the current representative.
            _, new_rep = min(candidates, key=lambda x: x[0])
            new_rid = new_rep.key
            new_representatives[species_key] = new_rid
            new_members[species_key] = [new_rid]
            unspeciated_genomes.remove(new_rid)

        # Partition population into species based on genetic similarity.
        while unspeciated_genomes:
            gid = unspeciated_genomes.pop()
            g = population[gid]

            # Find the species with the most similar representative.
            candidates = []
            for sid, rid in new_representatives.items():
                rep = population[rid]
                d = distances.get_distance(genome_0=rep, genome_1=g)
                if d < self.compatibility_threshold:
                    candidates.append((d, sid))

            if candidates:
                ignored_sdist, sid = min(candidates, key=lambda x: x[0])
                new_members[sid].append(gid)
            else:
                # No species is similar enough, create a new species, using
                # this genome as its representative.
                sid = next(self.indexer)
                new_representatives[sid] = gid
                new_members[sid] = [gid]

        # Update species collection based on new speciation.
        self.genome_to_species = {}
        for sid, rid in new_representatives.items():
            s = self.species.get(sid)
            if s is None:
                s = Specie(sid, generation)
                self.species[sid] = s

            members = new_members[sid]
            for gid in members:
                self.genome_to_species[gid] = sid

            member_dict = dict((gid, population[gid]) for gid in members)
            s.update(population[rid], member_dict)

        self.gdmean = distances.get_mean_distance()
        self.gdstdev = distances.get_std_distance()


class Specie:
    def __init__(self, key, generation):
        self.key = key
        self.created = generation
        self.last_improved = generation
        self.representative = None
        self.members = {}
        self.fitness = None
        self.adjusted_fitness = None
        self.fitness_history = []

    def update(self, representative, members):
        self.representative = representative
        self.members = members

    def get_fitnesses(self):
        return [m.fitness for m in self.members.values()]


class DistanceCalculation:
    def __init__(self):
        self.config = get_configuration()
        self.compatibility_weight_coefficient = self.config.compatibility_weight_coefficient
        self.compatibility_disjoint_coefficient = self.config.compatibility_disjoint_coefficient
        self.distances = {}
        self.hits = 0
        self.misses = 0

    def get_distance(self, genome_0: Genome, genome_1: Genome):
        key_0 = genome_0.key
        key_1 = genome_1.key

        distance = self.distances.get((key_0, key_1))
        if distance is None:
            distance = self._calculate_distance(genome_0, genome_1)
            self.distances[key_0, key_1] = distance
            self.distances[key_1, key_0] = distance
            self.misses += 1
        else:
            self.hits += 1

        return distance

    def get_mean_distance(self):
        distances_list = [d for d in self.distances.values()]
        return np.mean(distances_list)

    def get_std_distance(self):
        distances_list = [d for d in self.distances.values()]
        return np.std(distances_list)

    def _calculate_distance(self, genome_0: Genome, genome_1: Genome):
        """
                Returns the genetic distance between this genome and the other. This distance value
                is used to compute genome compatibility for speciation.
                """

        # Compute node gene distance component.
        node_distance = 0.0
        if genome_0.node_genes or genome_1.node_genes:
            disjoint_nodes = 0
            for k2 in genome_1.node_genes:
                if k2 not in genome_0.node_genes:
                    disjoint_nodes += 1

            for k1, n1 in genome_0.node_genes.items():
                n2 = genome_1.node_genes.get(k1)
                if n2 is None:
                    disjoint_nodes += 1
                else:
                    # Homologous genes compute their own distance value.
                    node_distance += self._node_distance(n1, n2)

            max_nodes = max(len(genome_0.node_genes), len(genome_1.node_genes))
            node_distance = (node_distance +
                             (self.compatibility_disjoint_coefficient *
                              disjoint_nodes)) / max_nodes

        # Compute connection gene differences.
        connection_distance = 0.0
        if genome_0.connection_genes or genome_1.connection_genes:
            disjoint_connections = 0
            for k2 in genome_1.connection_genes.keys():
                if k2 not in genome_0.connection_genes:
                    disjoint_connections += 1

            for k1, c1 in genome_0.connection_genes.items():
                c2 = genome_1.connection_genes.get(k1)
                if c2 is None:
                    disjoint_connections += 1
                else:
                    # Homologous genes compute their own distance value.
                    connection_distance += self._connection_distance(connection_1=c1, connection_2=c2)

            max_conn = max(len(genome_0.connection_genes), len(genome_1.connection_genes))
            connection_distance = (connection_distance +
                                   (self.compatibility_disjoint_coefficient *
                                    disjoint_connections)) / max_conn

        distance = node_distance + connection_distance
        return distance

    def _node_distance(self, node_1, node_2):
        '''
        Node distance is modified to account for both bias and standard deviation
        '''
        distance = l2_distance(v1=[node_1.bias_mean, node_1.bias_std],
                               v2=[node_2.bias_mean, node_2.bias_std])
        # TODO: add distance components if we are allowed to change activation or aggregation
        # if self.activation != other.activation:
        #     d += 1.0
        # if self.aggregation != other.aggregation:
        #     d += 1.0
        return distance * self.compatibility_weight_coefficient

    def _connection_distance(self, connection_1, connection_2):
        '''
        Connection distance is modified to account for both bias and standard deviation
        '''
        distance = l2_distance(v1=[connection_1.weight_mean, connection_1.weight_std],
                               v2=[connection_2.weight_mean, connection_2.weight_std])

        # this is not being used
        if connection_1.enabled != connection_2.enabled:
            distance += 1.0

        return distance * self.compatibility_weight_coefficient


def l2_distance(v1, v2):
    return np.sqrt(np.square(v1[0] + v2[0]) + np.square(v1[1] + v2[1]))
