import copy

import torch
from sklearn.model_selection import train_test_split
from torch.optim import Adam

from neat.evaluation.evaluate_simple import calculate_multinomial
from neat.evaluation.utils import _prepare_batch_data
from neat.fitness.kl_divergence import compute_kl_qw_pw
from neat.loss.vi_loss import get_loss
from neat.representation_mapping.genome_to_network.complex_stochastic_network import ComplexStochasticNetwork


class StandardTrainer:
    def __init__(self, dataset, n_epochs, n_output, problem_type, n_samples, beta, is_cuda, weight_decay=0.0005, lr=0.01):
        self.dataset = dataset
        self.is_cuda = is_cuda
        self.lr = lr
        self.weight_decay = weight_decay
        self.n_epochs = n_epochs
        self.n_output = n_output
        self.problem_type = problem_type
        self.n_samples = n_samples
        self.beta = beta

        self.network = None
        self.criterion = None
        self.optimizer = None
        self.final_loss = None
        self.best_loss_val = 10000
        self.best_network_state = None

    def train(self, genome):
        kl_qw_pw = compute_kl_qw_pw(genome=genome)
        # setup network
        self.network = ComplexStochasticNetwork(genome=genome, is_trainable=True, is_cuda=self.is_cuda)
        self.criterion = get_loss(problem_type=self.problem_type)
        if self.is_cuda:
            self.network.cuda()
            self.criterion.cuda()

        self.optimizer = Adam(self.network.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        x_batch, y_batch = self.dataset.x_train, self.dataset.y_train
        x_train, x_val, y_train, y_val = self.train_val_split(x_batch, y_batch,
                                                              problem_type=self.problem_type,
                                                              val_ratio=0.2)
        x_train, _ = _prepare_batch_data(x_batch=x_train,
                                         y_batch=y_train,
                                         problem_type=self.problem_type,
                                         is_gpu=False,    # this could be removed
                                         n_input=genome.n_input,
                                         n_output=genome.n_output,
                                         n_samples=self.n_samples)

        x_val, _ = _prepare_batch_data(x_batch=x_val,
                                       y_batch=y_val,
                                       problem_type=self.problem_type,
                                       is_gpu=False,
                                       n_input=genome.n_input,
                                       n_output=genome.n_output,
                                       n_samples=self.n_samples)

        if self.is_cuda:
            x_train = x_train.cuda()
            y_train = y_train.cuda()
            x_val = x_val.cuda()
            y_val = y_val.cuda()

        self.network.train()
        for epoch in range(self.n_epochs):
            loss_epoch = self._train_one(x_train, y_train, kl_qw_pw)
            if epoch % 10 == 0:
                _, _, _, loss_val = self._evaluate(x_val, y_val, network=self.network)

                if loss_val < self.best_loss_val:

                    self.best_loss_val = loss_val
                    self.best_network_state = copy.deepcopy(self.network.state_dict())
                    print(f'New best Val Loss: {loss_val}')

            if epoch % 200 == 0:
                print(f'Epoch = {epoch}. Training Loss: {loss_epoch}. '
                      f'Best Val. Loss: {self.best_loss_val}')
        self.network.clear_non_existing_weights(clear_grad=False)  # reset non-existing weights
        self.final_loss = loss_epoch
        print(f'Final Epoch = {epoch}. Training Error: {self.final_loss}')

    def _train_one(self, x_batch, y_batch, kl_qw_pw):
        # TODO: the kl_qw_pw returned by the network gives problems with backprop.
        output, _ = self.network(x_batch)
        output, _ = calculate_multinomial(output, self.n_samples, self.n_output)

        loss = self.criterion(y_pred=output, y_true=y_batch, kl_qw_pw=kl_qw_pw, beta=self.beta)
        loss_epoch = loss.data.item()
        self.optimizer.zero_grad()
        loss.backward()  # Backward Propagation
        # self.network.clear_non_existing_weights()  # zero_grad for those unexistent parameters
        self.optimizer.step()  # Optimizer update
        # self.network.clear_non_existing_weights(clear_grad=False)  # reset non-existing weights
        return loss_epoch

    def _evaluate(self, x_batch, y_batch, network):
        network.eval()

        chunks_x = []
        chunks_y_pred = []
        chunks_y_true = []

        with torch.no_grad():
            output, kl_qw_pw = network(x_batch)
            output, _ = calculate_multinomial(output, self.n_samples, self.n_output)

            # output, _, y_batch = _process_output_data(output, y_true=y_batch, n_samples=n_samples,
            #                                           n_output=genome.n_output, problem_type=problem_type, is_pass=is_pass)
            loss = self.criterion(y_pred=output, y_true=y_batch, kl_qw_pw=kl_qw_pw, beta=self.beta)
            # loss = self.criterion(output, y_batch)
            loss_epoch = loss.data.item()
            chunks_x.append(x_batch)
            chunks_y_pred.append(output)
            chunks_y_true.append(y_batch)

        x = torch.cat(chunks_x, dim=0)
        y_pred = torch.cat(chunks_y_pred, dim=0)
        y_true = torch.cat(chunks_y_true, dim=0)

        return x, y_true, y_pred, loss_epoch

    def train_val_split(self, x_batch, y_batch, problem_type, val_ratio=0.2):
        x_train, x_val, y_train, y_val = train_test_split(x_batch.numpy(), y_batch.numpy(),
                                                          test_size=val_ratio)
        x_train = torch.tensor(x_train).float()
        x_val = torch.tensor(x_val).float()
        if problem_type == 'classification':
            y_train = torch.tensor(y_train).long()
            y_val = torch.tensor(y_val).long()
        elif problem_type == 'regression':
            y_train = torch.tensor(y_train).float()
            y_val = torch.tensor(y_val).float()

        return x_train, x_val, y_train, y_val

    def get_best_network(self):
        network = ComplexStochasticNetwork(genome=self.network.genome, is_trainable=True)
        network.load_state_dict(self.best_network_state)
        return network
