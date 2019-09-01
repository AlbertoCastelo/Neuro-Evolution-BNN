import math

import torch
from torch.autograd import Variable
from torch.optim import Adam
from torch.utils.data import DataLoader

from deep_learning.probabilistic.feed_forward import ProbabilisticFeedForward
from neat.configuration import get_configuration
from neat.loss.vi_loss import _get_loss_by_problem, get_loss, get_beta
from deep_learning.standard.feed_forward import FeedForward


class EvaluateProbabilisticDL:

    def __init__(self, dataset, batch_size, n_samples, lr, weight_decay, n_epochs, n_neurons_per_layer, n_hidden_layers, is_cuda):
        self.config = get_configuration()
        self.is_cuda = is_cuda
        self.dataset = dataset
        self.batch_size = batch_size
        self.n_samples = n_samples
        self.lr = lr
        self.weight_decay = weight_decay
        self.n_epochs = n_epochs
        self.n_neurons_per_layer = n_neurons_per_layer
        self.n_hidden_layers = n_hidden_layers

    def run(self):
        self.dataset.generate_data()
        self.data_loader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True, num_workers=4)

        self.network = ProbabilisticFeedForward(n_input=self.config.n_input, n_output=self.config.n_output,
                                                n_neurons_per_layer=self.n_neurons_per_layer,
                                                n_hidden_layers=self.n_hidden_layers)

        self.criterion = get_loss(problem_type=self.config.problem_type)

        self.optimizer = Adam(self.network.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        if self.is_cuda:
            self.network.cuda()
            # self.criterion.cuda()

        self.m = math.ceil(len(self.data_loader) / self.batch_size)

        # train
        for epoch in range(self.n_epochs):
            loss_epoch = self.train_one(epoch)
            if epoch % 10 == 0:
                print(f'Epoch = {epoch}. Error: {loss_epoch}')

    def save_network(self, filename):
        # save weights
        torch.save(self.network.state_dict(), f'./../models/{filename}')

    def train_one(self, epoch):
        self.network.train()
        loss_epoch = 0

        for batch_idx, (x_batch, y_batch) in enumerate(self.data_loader):
            x_batch = x_batch.view(-1, self.config.n_input).repeat(self.n_samples, 1)
            y_batch = y_batch.view(-1, 1).repeat(self.n_samples, 1).squeeze()

            if self.is_cuda:
                x_batch = x_batch.cuda()
                y_batch = y_batch.cuda()
            # x_batch, y_batch = Variable(x_batch), Variable(y_batch)

            output, kl_qw_pw = self.network(x_batch)

            beta = get_beta(beta_type=self.config.beta_type, m=self.m, batch_idx=batch_idx, epoch=epoch,
                            n_epochs=self.n_epochs)
            kl_posterior = self.criterion(y_pred=output, y_true=y_batch, kl_qw_pw=kl_qw_pw, beta=beta)
            # print(kl_posterior.data.item())
            # loss_epoch += loss.data.item()

            self.optimizer.zero_grad()
            kl_posterior.backward()  # Backward Propagation
            self.optimizer.step()  # Optimizer update

            loss_epoch += kl_posterior.data.item()

        return loss_epoch

    def evaluate(self):
        self.network.eval()

        chunks_x = []
        chunks_y_pred = []
        chunks_y_true = []
        for x_batch, y_batch in self.data_loader:
            x_batch = x_batch.reshape((-1, self.config.n_input))
            x_batch = x_batch.view(-1, self.config.n_input).repeat(self.n_samples, 1)

            y_batch = y_batch.view(-1, 1).repeat(self.n_samples, 1).squeeze()

            if self.is_cuda:
                x_batch.cuda()
                y_batch.cuda()
            with torch.no_grad():
                y_pred, kl = self.network(x_batch)

                chunks_x.append(x_batch)
                chunks_y_pred.append(y_pred)
                chunks_y_true.append(y_batch)

        x = torch.cat(chunks_x, dim=0)
        y_pred = torch.cat(chunks_y_pred, dim=0)
        y_true = torch.cat(chunks_y_true, dim=0)

        return x, y_true, y_pred
