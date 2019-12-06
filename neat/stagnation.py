import sys
import numpy as np
from neat.configuration import get_configuration
from neat.utils import timeit


class Stagnation:
    '''
    Disclaimer: Taken from Python-NEAT project
    Updates species fitness history information,
        checking for ones that have not improved in max_stagnation generations,
        and - unless it would result in the number of species dropping below the configured
        species_elitism parameter if they were removed,
        in which case the highest-fitness species are spared -
        returns a list with stagnant species marked for removal.
    '''
    def __init__(self):
        self.config = get_configuration()

        self.species_fitness_function = self._get_species_fitness_function(function_name=
                                                                           self.config.species_fitness_function)
        self.species_elitism = self.config.species_elitism
        self.max_stagnation = self.config.max_stagnation

    @timeit
    def get_stagnant_species(self, species: dict, generation):

        species_data = []
        for sid, s in species.items():
            if s.fitness_history:
                prev_fitness = max(s.fitness_history)
            else:
                prev_fitness = -sys.float_info.max

            s.fitness = self.species_fitness_function(s.get_fitnesses())
            s.fitness_history.append(s.fitness)
            s.adjusted_fitness = None
            if prev_fitness is None or s.fitness > prev_fitness:
                s.last_improved = generation

            species_data.append((sid, s))

        # Sort in ascending fitness order.
        species_data.sort(key=lambda x: x[1].fitness)

        result = []
        species_fitnesses = []
        num_non_stagnant = len(species_data)
        for idx, (sid, s) in enumerate(species_data):
            # Override stagnant state if marking this species as stagnant would
            # result in the total number of species dropping below the limit.
            # Because species are in ascending fitness order, less fit species
            # will be marked as stagnant first.
            stagnant_time = generation - s.last_improved
            is_stagnant = False
            if num_non_stagnant > self.species_elitism:
                is_stagnant = stagnant_time >= self.max_stagnation

            if (len(species_data) - idx) <= self.species_elitism:
                is_stagnant = False

            if is_stagnant:
                num_non_stagnant -= 1

            result.append((sid, s, is_stagnant))
            species_fitnesses.append(s.fitness)

        return result

    def _get_species_fitness_function(self, function_name):
        if function_name == 'max':
            return np.max
        elif function_name == 'mean':
            return np.mean
        else:
            raise ValueError(f'Species Fitness Function not defined correctly: {function_name}')
