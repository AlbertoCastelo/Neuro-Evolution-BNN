import math

import torch
from torch import nn
from torch.nn import Parameter
import torch.nn.functional as F
from deep_learning.probabilistic.alternative_network.distributions import Normal, Normalout, distribution_selector
from neat.representation_mapping.genome_to_network.utils import get_activation


class BBBLinearFactorial(nn.Module):
    """
    Describes a Linear fully connected Bayesian layer with
    a distribution over each of the weights and biases
    in the layer.
    """
    def __init__(self, in_features, out_features, is_cuda, p_logvar_init=-3, p_pi=1.0, q_logvar_init=-5):
        # p_logvar_init, p_pi can be either
        # (list/tuples): prior model is a mixture of Gaussians components=len(p_pi)=len(p_logvar_init)
        # float: Gussian distribution
        # q_logvar_init: float, the approximate posterior is currently always a factorized gaussian
        super(BBBLinearFactorial, self).__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.is_cuda = is_cuda
        self.p_logvar_init = p_logvar_init
        self.q_logvar_init = q_logvar_init

        # Approximate posterior weights...
        self.qw_mean = Parameter(torch.Tensor(out_features, in_features))
        self.qw_logvar = Parameter(torch.Tensor(out_features, in_features))

        # optionally add bias
        # self.qb_mean = Parameter(torch.Tensor(out_features))
        # self.qb_logvar = Parameter(torch.Tensor(out_features))

        # ...and output...
        self.fc_qw_mean = Parameter(torch.Tensor(out_features, in_features))
        self.fc_qw_std = Parameter(torch.Tensor(out_features, in_features))

        # ...as normal distributions
        self.qw = Normal(mu=self.qw_mean, logvar=self.qw_logvar)
        # self.qb = Normal(mu=self.qb_mean, logvar=self.qb_logvar)
        self.fc_qw = Normalout(mu=self.fc_qw_mean, std=self.fc_qw_std)

        # initialise
        self.log_alpha = Parameter(torch.Tensor(1, 1))

        # prior model
        self.pw = distribution_selector(mu=0.0, logvar=p_logvar_init, pi=p_pi)
        # self.pb = distribution_selector(mu=0.0, logvar=p_logvar_init, pi=p_pi)

        # initialize all paramaters
        self.reset_parameters()

    def reset_parameters(self):
        # initialize (trainable) approximate posterior parameters
        stdv = 10. / math.sqrt(self.in_features)
        self.qw_mean.data.uniform_(-stdv, stdv)
        self.qw_logvar.data.uniform_(-stdv, stdv).add_(self.q_logvar_init)
        # self.qb_mean.data.uniform_(-stdv, stdv)
        # self.qb_logvar.data.uniform_(-stdv, stdv).add_(self.q_logvar_init)
        self.fc_qw_mean.data.uniform_(-stdv, stdv)
        self.fc_qw_std.data.uniform_(-stdv, stdv).add_(self.q_logvar_init)
        self.log_alpha.data.uniform_(-stdv, stdv)

    def forward(self, input):
        """
        Probabilistic forwarding method.
        :param input: data tensor
        :return: output, kl-divergence
        """

        fc_qw_mean = F.linear(input=input, weight=self.qw_mean)
        fc_qw_si = torch.sqrt(1e-8 + F.linear(input=input.pow(2), weight=torch.exp(self.log_alpha)*self.qw_mean.pow(2)))

        if self.is_cuda:
            fc_qw_mean.cuda()
            fc_qw_si.cuda()

        # sample from output
        if self.is_cuda:
            output = fc_qw_mean + fc_qw_si * (torch.randn(fc_qw_mean.size())).cuda()
        else:
            output = fc_qw_mean + fc_qw_si * (torch.randn(fc_qw_mean.size()))

        if self.is_cuda:
            output.cuda()

        w_sample = self.fc_qw.sample()

        # KL divergence
        qw_logpdf = self.fc_qw.logpdf(w_sample)

        kl = torch.sum(qw_logpdf - self.pw.logpdf(w_sample))

        return output, kl

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class BBBLinear(nn.Module):
    """
    Describes a Linear fully connected Bayesian layer with
    a distribution over each of the weights and biases
    in the layer.
    """
    def __init__(self, in_features, out_features, is_cuda, p_logvar_init=-3, p_pi=1.0, q_logvar_init=-5):
        # p_logvar_init, p_pi can be either
        # (list/tuples): prior model is a mixture of Gaussians components=len(p_pi)=len(p_logvar_init)
        # float: Gussian distribution
        # q_logvar_init: float, the approximate posterior is currently always a factorized gaussian
        super(BBBLinear, self).__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.is_cuda = is_cuda
        self.p_logvar_init = p_logvar_init
        self.q_logvar_init = q_logvar_init

        self.device = torch.device("cuda:0" if self.is_cuda else "cpu")


        self.W_mu = Parameter(torch.empty((out_features, in_features), device=self.device))
        self.W_rho = Parameter(torch.empty((out_features, in_features), device=self.device))

        self.bias_mu = Parameter(torch.empty((out_features,), device=self.device))
        self.bias_rho = Parameter(torch.empty((out_features,), device=self.device))
        # optionally add bias
        # self.qb_mean = Parameter(torch.Tensor(out_features))
        # self.qb_logvar = Parameter(torch.Tensor(out_features))

        # initialize all paramaters
        self.reset_parameters()

    def reset_parameters(self):
        self.W_mu.data.normal_(*(0, 0.1))
        self.W_rho.data.normal_(*(-3, 0.1))

        self.bias_mu.data.normal_(*(0, 0.1))
        self.bias_rho.data.normal_(*(-3, 0.1))

        # initialize (trainable) approximate posterior parameters
        # stdv = 10. / math.sqrt(self.in_features)
        # self.qw_mean.data.uniform_(-stdv, stdv)
        # self.qw_logvar.data.uniform_(-stdv, stdv).add_(self.q_logvar_init)
        # # self.qb_mean.data.uniform_(-stdv, stdv)
        # # self.qb_logvar.data.uniform_(-stdv, stdv).add_(self.q_logvar_init)
        # self.fc_qw_mean.data.uniform_(-stdv, stdv)
        # self.fc_qw_std.data.uniform_(-stdv, stdv).add_(self.q_logvar_init)
        # self.log_alpha.data.uniform_(-stdv, stdv)

    def forward(self, input):
        """
        Probabilistic forwarding method.
        :param input: data tensor
        :return: output, kl-divergence
        """
        W_eps = torch.empty(self.W_mu.size()).normal_(0, 1).to(self.device)
        self.W_sigma = torch.log1p(torch.exp(self.W_rho))
        weight = self.W_mu + W_eps * self.W_sigma

        bias_eps = torch.empty(self.bias_mu.size()).normal_(0, 1).to(self.device)
        self.bias_sigma = torch.log1p(torch.exp(self.bias_rho))
        bias = self.bias_mu + bias_eps * self.bias_sigma

        output = F.linear(input, weight, bias)
        kl = 0.0

        return output, kl

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


Linear = BBBLinearFactorial


class ProbabilisticFeedForwardAlternative(nn.Module):
    def __init__(self, n_input, n_output, is_cuda, n_neurons_per_layer=3, n_hidden_layers=2):
        super(ProbabilisticFeedForwardAlternative, self).__init__()
        self.n_input = n_input
        self.n_output = n_output
        self.n_neurons_per_layer = n_neurons_per_layer
        self.n_hidden_layers = n_hidden_layers
        self.is_cuda = is_cuda
        self.activation = get_activation()
        self.n_layers = n_hidden_layers
        in_features = n_input

        # hidden layers
        for i in range(n_hidden_layers, 0, -1):
            layer = Linear(in_features=in_features, out_features=n_neurons_per_layer,
                                       is_cuda=is_cuda)
            setattr(self, f'layer_{i}', layer)
            setattr(self, f'activation_{i}', self.activation)
            in_features = n_neurons_per_layer

        layer = Linear(in_features=in_features, out_features=n_output, is_cuda=is_cuda)
        setattr(self, f'layer_0', layer)

    def reset_parameters(self):
        # hidden layers
        for i in range(self.n_hidden_layers, -1, -1):
            getattr(self, f'layer_{i}').reset_parameters()

    def forward(self, x):
        kl_qw_pw = 0.0
        start_index = self.n_layers
        for i in range(start_index, -1, -1):
            x, kl_layer = getattr(self, f'layer_{i}')(x)
            kl_qw_pw += kl_layer
            if i > 0:
                x = getattr(self, f'activation_{i}')(x)
        return x, kl_qw_pw
