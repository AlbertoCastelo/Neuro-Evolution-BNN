import torch
from torch.optim import Adam
from torch.utils.data import DataLoader

from neat.configuration import get_configuration
from neat.loss.vi_loss import _get_loss_by_problem
from tests_non_automated.deep_learning.feed_forward import FeedForward


class EvaluateStandardDL:

    def __init__(self, dataset, batch_size, lr, weight_decay, n_epochs, n_neurons_per_layer, n_hidden_layers, is_cuda):
        self.config = get_configuration()
        self.is_cuda = is_cuda
        self.dataset = dataset
        self.batch_size = batch_size
        self.lr = lr
        self.weight_decay = weight_decay
        self.n_epochs = n_epochs
        self.n_neurons_per_layer = n_neurons_per_layer
        self.n_hidden_layers = n_hidden_layers

    def run(self):
        self.dataset.generate_data()
        self.data_loader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True, num_workers=4)

        self.network = FeedForward(n_input=self.config.n_input, n_output=self.config.n_output,
                                   n_neurons_per_layer=self.n_neurons_per_layer,
                                   n_hidden_layers=self.n_hidden_layers)

        self.criterion = _get_loss_by_problem(problem_type=self.config.problem_type)

        self.optimizer = Adam(self.network.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        if self.is_cuda:
            self.network.cuda()
            self.criterion.cuda()

        # train
        for epoch in range(self.n_epochs):
            loss_epoch = self.train_one()
            if epoch % 10 == 0:
                print(f'Epoch = {epoch}. Error: {loss_epoch}')

    def save_network(self, filename):
        # save weights
        torch.save(self.network.state_dict(), f'./models/{filename}')

    def train_one(self):

        self.network.train()
        loss_epoch = 0
        for x_batch, y_batch in self.data_loader:
            # x_batch = x_batch.reshape((-1, config.n_input))
            # x_batch = x_batch.float()
            # # print(x_batch.shape)
            # y_batch = y_batch.long()
            # print(y_batch.shape)
            # forward pass
            if self.is_cuda:
                x_batch.cuda()
                y_batch.cuda()
            output = self.network(x_batch)
            loss = self.criterion(output, y_batch)
            loss_epoch += loss.data.item()

            self.optimizer.zero_grad()
            loss.backward()  # Backward Propagation
            self.optimizer.step()  # Optimizer update

        return loss_epoch

    def predict(self, x_pred):
        self.network.eval()

        x_batch = x_pred.reshape((-1, self.config.n_input))
        x_batch = x_batch.float()

        # forward pass
        with torch.no_grad():
            y_pred = self.network(x_batch)
        return y_pred
