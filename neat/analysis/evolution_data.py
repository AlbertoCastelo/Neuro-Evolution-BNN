import pandas as pd


class EvolutionData:
    def __init__(self, report, execution_id):
        self.report = report
        self.execution_id = execution_id

        self.fitness_evolution = None
        self.fitness_by_specie = None

    def process(self):
        generation_metrics = self.report.data['generation_metrics']
        n_generations = len(generation_metrics)
        fitness_evolution = []

        fitness_by_specie = []
        n_species = 5
        for gen in range(n_generations):
            best_fitness = generation_metrics[str(gen)]['best_individual_fitness']
            worst_fitness = generation_metrics[str(gen)]['min_fitness']
            mean_fitness = generation_metrics[str(gen)]['mean_fitness']
            fitness_evolution.append([gen, best_fitness, worst_fitness, mean_fitness])

            fitness_by_specie_gen = generation_metrics[str(gen)]['genomes_fitness_per_specie']
            mean_fitness_specie_list = [gen]
            for specie, fitnesses_per_specie in fitness_by_specie_gen.items():
                mean_fitness_specie_list.append(np.max(fitnesses_per_specie))

            fitness_by_specie.append(mean_fitness_specie_list)
        self.fitness_evolution = pd.DataFrame(fitness_evolution,
                                              columns=['generation', 'best_fitness', 'worst_fitness', 'mean_fitness'])

        self.fitness_by_specie = pd.DataFrame(fitness_by_specie,
                                              columns=['generation'] + [f'specie-{i + 1}' for i in range(n_species)])
