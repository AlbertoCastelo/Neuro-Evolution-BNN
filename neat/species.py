from itertools import count
import numpy as np

from experiments.logger import logger
from neat.configuration import get_configuration
from neat.gene import NodeGene, ConnectionGene
from neat.genome import Genome
from neat.utils import timeit


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

        self.gdmean = None
        self.gdstdev = None

    @timeit
    def speciate(self, population: dict, generation: int):
        """
        Disclaimer: code copied from NEAT-Python: https://neat-python.readthedocs.io/en/latest/

        Place genomes into species by genetic similarity.

        Note that this method assumes the current representatives of the species are from the old
        generation, and that after speciation has been performed, the old representatives should be
        dropped and replaced with representatives from the new generation.  If you violate this
        assumption, you should make sure other necessary parts of the code are updated to reflect
        the new behavior.
        """
        if generation > 0 and len(self.species) == 0:
            return ValueError('All species have died')
        
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

            if self._enough_candidates(candidates):
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
                logger.debug(f'New specie')
                s = Specie(sid, generation)
                self.species[sid] = s

            members = new_members[sid]
            for gid in members:
                self.genome_to_species[gid] = sid

            member_dict = dict((gid, population[gid]) for gid in members)
            s.update(population[rid], member_dict)

        self.gdmean = distances.get_mean_distance()
        self.gdstdev = distances.get_std_distance()
        logger.debug(f'Number of species: {len(self.species)}')

    def _enough_candidates(self, candidates):
        return len(candidates) > 0


class FixSpeciationEngine:
    '''
    Speciation in NEAT
    https://apps.cs.utexas.edu/tech_reports/reports/tr/TR-1972.pdf
    '''

    def __init__(self):
        self.config = get_configuration()
        self.indexer = count(1)
        self.n_species = self.config.n_species
        self.species = {}
        self.genome_to_species = {}

        self.compatibility_threshold = self.config.compatibility_threshold

        self.gdmean = None
        self.gdstdev = None

    @timeit
    def speciate(self, population: dict, generation: int):
        """
        Disclaimer: code copied from NEAT-Python: https://neat-python.readthedocs.io/en/latest/

        Place genomes into species by genetic similarity.

        Note that this method assumes the current representatives of the species are from the old
        generation, and that after speciation has been performed, the old representatives should be
        dropped and replaced with representatives from the new generation.  If you violate this
        assumption, you should make sure other necessary parts of the code are updated to reflect
        the new behavior.
        """
        if len(self.species) == 0:
            self.species = self._generate_initial_species(population)
            return

        if len(self.species) == 0:
            return ValueError('All species have died')

        unspeciated_genomes = list(population.keys())
        distances = DistanceCalculation()
        new_representatives = {}
        new_members = {}

        # using last generation species it search for the distance between the genomes in the new population and the
        # representative from the specie
        self._define_new_representatives(distances, new_members, new_representatives, population, unspeciated_genomes)

        # fill species if some have died: look at initialization
        while len(self.species) < self.n_species:
            # create more species
            pass

        # Partition population into species based on genetic similarity.
        new_representatives, new_members = \
            self._assign_genome_to_specie(distances, new_members, new_representatives, population, unspeciated_genomes)


        # Update species collection based on new speciation.
        self.species = self._members_to_species(self.species, new_members, new_representatives, population)

        self.gdmean = distances.get_mean_distance()
        self.gdstdev = distances.get_std_distance()
        logger.debug(f'Number of species: {len(self.species)}')

    def _define_new_representatives(self, distances, new_members, new_representatives, population, unspeciated_genomes):
        for species_key, specie in self.species.items():
            candidates = []
            for genome_key in unspeciated_genomes:
                genome = population[genome_key]
                d = distances.get_distance(genome_0=specie.representative, genome_1=genome)
                candidates.append((d, genome))

            if self._enough_candidates(candidates):
                # The new representative is the genome closest to the current representative.
                _, new_rep = min(candidates, key=lambda x: x[0])
                new_rid = new_rep.key
                new_representatives[species_key] = new_rid
                new_members[species_key] = [new_rid]
                unspeciated_genomes.remove(new_rid)

    def _enough_candidates(self, candidates):
        return len(candidates) > 0

    @timeit
    def _generate_initial_species(self, population):

        unspeciated_genomes = list(population.keys())
        distances = DistanceCalculation()
        new_representatives = {}
        new_members = {}
        total_distance_by_genome = np.zeros(shape=(len(unspeciated_genomes), len(unspeciated_genomes)))

        for i, genome_0_key in enumerate(unspeciated_genomes):
            for j, genome_1_key in enumerate(unspeciated_genomes):
                if j > i:
                    d = distances.get_distance(genome_0=population[genome_0_key],
                                               genome_1=population[genome_1_key])
                    total_distance_by_genome[i, j] = d
                    total_distance_by_genome[j, i] = d
        index_top_farthest_genomes = total_distance_by_genome.sum(1).argsort()[-self.n_species:][::-1]
        print(index_top_farthest_genomes)
        print(len(unspeciated_genomes))
        # create new species
        for index_genome in index_top_farthest_genomes:
            genome_key = unspeciated_genomes[index_genome]
            sid = next(self.indexer)
            new_representatives[sid] = genome_key
            new_members[sid] = [genome_key]
            unspeciated_genomes.remove(genome_key)
            # np.delete(total_distance_by_genome, index_genome, axis=1)

        new_representatives, new_members = \
            self._assign_genome_to_specie(distances, new_members, new_representatives, population, unspeciated_genomes)

        species = {}
        species = self._members_to_species(species, new_members, new_representatives, population)
        return species

    def _members_to_species(self, species, new_members, new_representatives, population):

        for specie_key, genome_key in new_representatives.items():
            specie = species.get(specie_key)
            if specie is None:
                specie = Specie(key=specie_key, generation=0)
            genome_representative = population[new_representatives[specie_key]]
            members = dict((gid, population[gid]) for gid in new_members[specie_key])
            specie.update(representative=genome_representative,
                          members=members)
            species[specie_key] = specie
        return species

    def _assign_genome_to_specie(self, distances, new_members, new_representatives, population, unspeciated_genomes):
        while unspeciated_genomes:
            genome_0_key = unspeciated_genomes.pop()
            # for i, genome_0_key in enumerate(unspeciated_genomes):
            min_distance = np.inf
            min_specie_key = None
            for sid, specie_genome_key in new_representatives.items():
                d = distances.get_distance(population[specie_genome_key], population[genome_0_key])
                if d < min_distance:
                    min_distance = d
                    min_specie_key = sid

            new_members[min_specie_key].append(genome_0_key)
            # unspeciated_genomes.remove(genome_0_key)
        return new_representatives, new_members


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

    def get_distance_from_keys(self, genome_key_0, genome_key_1):
        distance = self.distances.get((genome_key_0, genome_key_1))
        if distance is None:
            raise ValueError('Distance not available')
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

    def _node_distance(self, node_1: NodeGene, node_2: NodeGene):
        '''
        Node distance is modified to account for both bias and standard deviation
        '''
        distance = l2_distance(v1=[node_1.get_mean(), node_1.get_std()],
                               v2=[node_2.get_mean(), node_2.get_std()])
        # TODO: add distance components if we are allowed to change activation or aggregation
        # if self.activation != other.activation:
        #     d += 1.0
        # if self.aggregation != other.aggregation:
        #     d += 1.0
        return distance * self.compatibility_weight_coefficient

    def _connection_distance(self, connection_1: ConnectionGene, connection_2: ConnectionGene):
        '''
        Connection distance is modified to account for both bias and standard deviation
        '''
        distance = l2_distance(v1=[connection_1.get_mean(), connection_1.get_std()],
                               v2=[connection_2.get_mean(), connection_2.get_std()])

        # this is not being used
        if connection_1.enabled != connection_2.enabled:
            distance += 1.0

        return distance * self.compatibility_weight_coefficient


def l2_distance(v1, v2):
    return np.sqrt(np.square(v1[0] + v2[0]) + np.square(v1[1] + v2[1]))
