import copy
import os
import torch
from sklearn.model_selection import train_test_split
from torch.optim import Adam
from deep_learning.probabilistic.feed_forward import ProbabilisticFeedForward
from neat.configuration import get_configuration
from neat.evaluation.utils import _prepare_batch_data
from neat.loss.vi_loss import get_loss


class EvaluateProbabilisticDL:

    def __init__(self, dataset, batch_size, n_samples, lr, weight_decay, n_epochs, n_neurons_per_layer, n_hidden_layers,
                 is_cuda, beta):
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
        self.beta = beta
        self.best_loss_val = 10000
        self.best_network_state = None

    def run(self):

        self._initialize()

        if self.is_cuda:
            self.network.cuda()
            self.criterion.cuda()

        x_batch, y_batch = self.dataset.x_train, self.dataset.y_train
        x_train, x_val, y_train, y_val = self.train_val_split(x_batch, y_batch, val_ratio=0.2)

        x_train, y_train = _prepare_batch_data(x_batch=x_train,
                                               y_batch=y_train,
                                               problem_type=self.config.problem_type,
                                               is_gpu=self.config.is_gpu,
                                               n_input=self.config.n_input,
                                               n_output=self.config.n_output,
                                               n_samples=self.config.n_samples)
        x_val, y_val = _prepare_batch_data(x_batch=x_val,
                                           y_batch=y_val,
                                           problem_type=self.config.problem_type,
                                           is_gpu=self.config.is_gpu,
                                           n_input=self.config.n_input,
                                           n_output=self.config.n_output,
                                           n_samples=self.config.n_samples)

        # train
        for epoch in range(self.n_epochs):
            loss_train = self._train_one(x_train, y_train)
            if epoch % 10 == 0:
                print(f'Epoch = {epoch}. Error: {loss_train}')
                _, _, _, loss_val = self._evaluate(x_val, y_val, network=self.network)

                if loss_val < self.best_loss_val:

                    self.best_loss_val = loss_val
                    self.best_network_state = copy.deepcopy(self.network.state_dict())
                    print(f'New best network: {loss_val}')

    def _initialize(self):
        self.dataset.generate_data()
        self.network = ProbabilisticFeedForward(n_input=self.config.n_input, n_output=self.config.n_output,
                                                is_cuda=self.is_cuda,
                                                n_neurons_per_layer=self.n_neurons_per_layer,
                                                n_hidden_layers=self.n_hidden_layers)
        self.criterion = get_loss(problem_type=self.config.problem_type)
        self.optimizer = Adam(self.network.parameters(), lr=self.lr, weight_decay=self.weight_decay)

    def _train_one(self, x_batch, y_batch):
        if self.is_cuda:
            x_batch.cuda()
            y_batch.cuda()
        output, kl_qw_pw = self.network(x_batch)
        # output, _, y_batch = _process_output_data(output, y_true=y_batch, n_samples=n_samples,
        #                                           n_output=genome.n_output, problem_type=problem_type, is_pass=is_pass)
        loss = self.criterion(y_pred=output, y_true=y_batch, kl_qw_pw=kl_qw_pw, beta=self.beta)
        # loss = self.criterion(output, y_batch)
        loss_epoch = loss.data.item()
        self.optimizer.zero_grad()
        loss.backward()  # Backward Propagation
        self.optimizer.step()  # Optimizer update
        return loss_epoch

    def save_network(self, name):
        # save weights
        filename = ''.join([os.path.dirname(os.path.realpath(__file__)), f'/../models/{name}'])
        torch.save(self.network.state_dict(), filename)

    def evaluate(self):
        x_batch, y_batch = self.dataset.x_test, self.dataset.y_test
        x_batch, y_batch = _prepare_batch_data(x_batch=x_batch,
                                               y_batch=y_batch,
                                               problem_type=self.config.problem_type,
                                               is_gpu=self.config.is_gpu,
                                               n_input=self.config.n_input,
                                               n_output=self.config.n_output,
                                               n_samples=self.config.n_samples)

        network = ProbabilisticFeedForward(n_input=self.config.n_input, n_output=self.config.n_output,
                                           is_cuda=self.is_cuda,
                                           n_neurons_per_layer=self.n_neurons_per_layer,
                                           n_hidden_layers=self.n_hidden_layers)
        network.load_state_dict(self.best_network_state)

        x, y_true, y_pred, loss = self._evaluate(x_batch, y_batch, network=network)

        return x, y_true, y_pred

    def _evaluate(self, x_batch, y_batch, network):
        network.eval()

        chunks_x = []
        chunks_y_pred = []
        chunks_y_true = []

        if self.is_cuda:
            x_batch.cuda()
            y_batch.cuda()
        with torch.no_grad():
            output, kl_qw_pw = network(x_batch)
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

    def train_val_split(self, x_batch, y_batch, val_ratio=0.2):
        x_train, x_val, y_train, y_val = train_test_split(x_batch.numpy(), y_batch.numpy(),
                                                          test_size=val_ratio)
        x_train = torch.tensor(x_train).float()
        x_val = torch.tensor(x_val).float()
        y_train = torch.tensor(y_train).long()
        y_val = torch.tensor(y_val).long()

        return x_train, x_val, y_train, y_val
