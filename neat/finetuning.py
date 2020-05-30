from neat.evaluation.evaluation_engine import EvaluationStochasticEngine
from neat.evaluation.utils import get_dataset
from neat.genome import Genome
from neat.representation_mapping.network_to_genome.stochastic_network_to_genome import \
    convert_stochastic_network_to_genome
from neat.standard_training.standard_trainer import StandardTrainer


class FineTuner:
    '''
    Given a population or species, it will fine tune the most promising models and select the best
    '''
    def __init__(self, species, config, only_best=True, is_cuda=False, weight_decay=0.0005, lr=0.01):
        self.species = species
        self.config = config
        self.only_best = only_best
        self.is_cuda = is_cuda
        self.n_epochs = config.epochs_fine_tuning
        self.weight_decay = weight_decay
        self.lr = lr

        self.species_best_genome = {}

    def run(self):
        if self.only_best:
            self._finetune_best()
        else:
            self._finetune_best_per_specie()

    def _finetune_best_per_specie(self):
        for specie_key, specie in self.species.items():
            print(f'Fine-tuning: {specie_key}')
            # genome_to_finetune = specie.representative
            genome_to_finetune = specie.get_fittest()
            self.species_best_genome[specie_key] = self._finetune_genome(genome_to_finetune)
        self.config.parallel_evaluation = False
        evaluation_engine = EvaluationStochasticEngine()
        self.species_best_genome = evaluation_engine.evaluate(population=self.species_best_genome)

    def _finetune_best(self):
        best_genome = None
        best_fitness = -100000
        for specie_key, specie in self.species.items():
            best_genome_at_specie = specie.get_fittest()
            if best_genome_at_specie.fitness > best_fitness:
                best_genome = best_genome_at_specie
                best_fitness = best_genome_at_specie.fitness
        best_genome_finetuned = self._finetune_genome(best_genome)
        self.config.parallel_evaluation = False
        evaluation_engine = EvaluationStochasticEngine()
        self.species_best_genome = evaluation_engine.evaluate(population={1: best_genome_finetuned})


    def _finetune_genome(self, genome: Genome):

        stg_trainer = StandardTrainer(dataset=self._get_dataset(),
                                      n_samples=self.config.n_samples,
                                      n_output=self.config.n_output,
                                      problem_type=self.config.problem_type,
                                      beta=self.config.beta,
                                      n_epochs=self.n_epochs, is_cuda=self.is_cuda,
                                      weight_decay=self.weight_decay, lr=self.lr)
        stg_trainer.train(genome)
        network = stg_trainer.get_best_network()

        return convert_stochastic_network_to_genome(network=network, original_genome=genome,
                                                    fitness=-stg_trainer.best_loss_val,
                                                    fix_std=genome.genome_config.fix_std)

    def _get_dataset(self):
        return get_dataset(dataset=self.config.dataset, train_percentage=self.config.train_percentage,
                           random_state=self.config.dataset_random_state, noise=self.config.noise)
