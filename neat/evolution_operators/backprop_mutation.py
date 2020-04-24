from torch.optim import Adam

from neat.evaluation.utils import _prepare_batch_data
from neat.evolution_operators.mutation import Mutation
from neat.fitness.kl_divergence import compute_kl_qw_pw
from neat.genome import Genome
from neat.loss.vi_loss import get_loss
from neat.representation_mapping.genome_to_network.complex_stochastic_network import ComplexStochasticNetwork
from neat.representation_mapping.network_to_genome.stochastic_network_to_genome import \
    convert_stochastic_network_to_genome

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
        return convert_stochastic_network_to_genome(network=self.network, original_genome=genome)

    def _mutate(self, genome: Genome):
        kl_qw_pw = compute_kl_qw_pw(genome=genome)
        # setup network
        self.network = ComplexStochasticNetwork(genome=genome, is_trainable=True)
        self.loss = get_loss(problem_type=self.problem_type)
        self.optimizer = Adam(self.network.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        x_batch, y_batch = self.dataset.x_train, self.dataset.y_train
        x_batch, y_batch = _prepare_batch_data(x_batch=x_batch,
                                               y_batch=y_batch,
                                               problem_type=self.problem_type,
                                               is_gpu=False,
                                               n_input=genome.n_input,
                                               n_output=genome.n_output,
                                               n_samples=self.n_samples)
        # train
        self.network.train()
        for epoch in range(self.n_epochs):
            loss_epoch = 0.0

            output, _ = self.network(x_batch)

            loss = self.loss(y_pred=output, y_true=y_batch, kl_qw_pw=kl_qw_pw, beta=self.beta)
            loss_epoch += loss.data.item()

            self.optimizer.zero_grad()
            loss.backward()  # Backward Propagation
            # self.network.clear_non_existing_weights()  # zero_grad for those unexistent parameters
            self.optimizer.step()  # Optimizer update
            # self.network.clear_non_existing_weights(clear_grad=False)  # reset non-existing weights
            if epoch % 10 == 0:
                print(f'Epoch = {epoch}. Error: {loss_epoch}')

        self.final_loss = float(loss.data)
