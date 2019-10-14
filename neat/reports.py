import numpy as np
from neat.configuration import write_json_file_from_dict


class EvolutionReport:

    def __init__(self, experiment_name: str):
        self.experiment_name = experiment_name
        # datetime = datetime.datetime.now()
        datetime = None
        self.execution_id = f'{experiment_name}_{datetime}'
        self.generation_metrics = dict()
        self.best_individual = None

    def report_new_generation(self, generation: int, population: dict):
        best_individual_key = -1
        best_individual_fitness = -1000000
        fitness_all = []
        for key, genome in population.items():
            fitness_all.append(genome.fitness)
            if genome.fitness > best_individual_fitness:
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

        if self.best_individual is None or self.best_individual.fitness < best_individual_fitness:
            self.best_individual = population.get(best_individual_key)
            print(f'    New best individual found:{round(self.best_individual.fitness, 3)}')

    def generate_final_report(self):
        best_individual = self.get_best_individual().to_dict()
        filename = f'./executions/{self.execution_id}.json'

        write_json_file_from_dict(data=best_individual, filename=filename)

    def persist(self):
        pass

    def get_best_individual(self):
        return self.best_individual
