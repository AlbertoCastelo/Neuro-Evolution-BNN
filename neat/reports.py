import numpy as np


class EvolutionReport:

    def __init__(self):
        self.generation_metrics = dict()
        self.best_individual = None

    def report_new_generation(self, generation, population):
        best_individual_key = -1
        best_individual_fitness = 1000000
        fitness_all = []
        for key, genome in population.items():
            fitness_all.append(genome.fitness)
            if genome.fitness < best_individual_fitness:
                best_individual_fitness = genome.fitness
                best_individual_key = genome.key

        data = {'best_individual_fitness': best_individual_fitness,
                'best_individual_key': best_individual_key,
                'all_fitness': fitness_all,
                'min': round(min(fitness_all), 3),
                'max': round(max(fitness_all), 3),
                'mean': round(np.mean(fitness_all), 3)}
        print(f'Generation {generation}. Best fitness: {round(max(fitness_all), 3)}. '
              f'Mean fitness: {round(np.mean(fitness_all), 3)}')
        self.generation_metrics[generation] = data

    def generate_final_report(self):
        pass

    def persist(self):
        pass

    def get_best_individual(self):
        return self.best_individual
