from neat.evolution_operators.mutation import Mutation
from neat.genome import Genome
from neat.representation_mapping.network_to_genome.stochastic_network_to_genome import \
    convert_stochastic_network_to_genome
from neat.standard_training.standard_trainer import StandardTrainer

BACKPROP_MUTATION = 'backprop_mutation'


class BackPropMutation(Mutation):
    def __init__(self, dataset, n_samples, problem_type, beta, n_epochs, weight_decay=0.0005, lr=0.01):
        self.dataset = dataset
        self.lr = lr
        self.weight_decay = weight_decay
        self.n_epochs = n_epochs
        self.problem_type = problem_type
        self.n_samples = n_samples
        self.beta = beta


        self.network = None

    def mutate(self, genome: Genome):
        if self.network is None:
            self._mutate(genome)
        return convert_stochastic_network_to_genome(network=self.network, original_genome=genome,
                                                    fitness=-self.final_loss, fix_std=genome.genome_config.fix_std)

    def _mutate(self, genome: Genome):
        stg_trainer = StandardTrainer(dataset=self.dataset, n_samples=self.n_samples, problem_type=self.problem_type,
                                      beta=self.beta,
                                      n_epochs=self.n_epochs, weight_decay=self.weight_decay, lr=self.lr)
        stg_trainer.train(genome)
        self.network = stg_trainer.network
        self.final_loss = stg_trainer.final_loss
