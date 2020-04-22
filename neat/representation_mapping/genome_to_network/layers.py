import math

from torch import nn
from torch.distributions import Normal, kl_divergence
from torch.nn import Parameter
import torch
import torch.nn.functional as F


class StochasticLinearParameters:
    @staticmethod
    def create(qw_mean, qw_logvar, qb_mean, qb_logvar):
        return StochasticLinearParameters(qw_mean, qw_logvar, qb_mean, qb_logvar)

    def __init__(self, qw_mean, qw_logvar, qb_mean, qb_logvar, log_alpha=None):
        self.qw_mean = qw_mean
        self.qw_logvar = qw_logvar
        self.qb_mean = qb_mean
        self.qb_logvar = qb_logvar
        self.log_alpha = log_alpha


class StochasticLinear(nn.Module):

    def __init__(self, in_features, out_features, is_cuda=False, parameters: StochasticLinearParameters = None,
                 n_samples=10, q_logvar_init=-5, is_trainable=False):
        # p_logvar_init, p_pi can be either
        # (list/tuples): prior model is a mixture of Gaussians components=len(p_pi)=len(p_logvar_init)
        # float: Gussian distribution
        # q_logvar_init: float, the approximate posterior is currently always a factorized gaussian
        super(StochasticLinear, self).__init__()

        self.is_trainable = is_trainable
        self.in_features = in_features
        self.out_features = out_features
        self.is_cuda = is_cuda
        self.q_logvar_init = q_logvar_init
        self.n_samples = n_samples

        self.qw_mean = None
        self.qw_logvar = None
        self.qb_mean = None
        self.qb_logvar = None
        self.log_alpha = None

        if parameters is None:
            # Approximate posterior weights and biases
            self.qw_mean = Parameter(torch.Tensor(out_features, in_features))
            self.qw_logvar = Parameter(torch.Tensor(out_features, in_features))

            # optionally add bias
            self.qb_mean = Parameter(torch.Tensor(out_features))
            self.qb_logvar = Parameter(torch.Tensor(out_features))

            self.log_alpha = Parameter(torch.Tensor(1, 1))

            # initialize all paramaters
            self.reset_parameters()
        else:
            # this parameters are known
            # print(f'qw_mean: {parameters.qw_mean.shape}')
            # print(f'qw_logvar: {parameters.qw_logvar.shape}')
            # print(f'qb_mean: {parameters.qb_mean.shape}')
            # print(f'qb_logvar: {parameters.qb_logvar.shape}')
            self.qw_mean = parameters.qw_mean
            self.qw_logvar = parameters.qw_logvar

            self.qb_mean = parameters.qb_mean
            self.qb_logvar = parameters.qb_logvar

            if parameters.log_alpha is not None:
                self.log_alpha = parameters.log_alpha

            if self.is_trainable:
                self.qw_mean = Parameter(self.qw_mean)
                self.qw_logvar = Parameter(self.qw_logvar)

                self.qb_mean = Parameter(self.qb_mean)
                self.qb_logvar = Parameter(self.qb_logvar)
                self.log_alpha = Parameter(self.log_alpha)


    def reset_parameters(self):
        # initialize (trainable) approximate posterior parameters
        stdv = 10. / math.sqrt(self.in_features)
        self.qw_mean.data.uniform_(-stdv, stdv)
        self.qw_logvar.data.uniform_(-stdv, stdv).add_(self.q_logvar_init)
        self.qb_mean.data.uniform_(-stdv, stdv)
        self.qb_logvar.data.uniform_(-stdv, stdv).add_(self.q_logvar_init)

        # assumes 1 sigma for all weights per layer.
        self.log_alpha.data.uniform_(-stdv, stdv)

    def forward_2(self, x):
        # EQUATION
        # y = x·(mu_w + exp(log_var_w + 1.0)·N(0,1)) +
        #     (mu_b + + exp(log_var_b + 1.0)·N(0,1))
        batch_size = x.shape[0]

        print(f'x: {x.shape}')
        qw_var = torch.exp(1.0 + self.qw_logvar)
        qb_var = torch.exp(1.0 + self.qb_logvar)

        w_sample_size = (self.out_features * self.n_samples, self.in_features)
        qw_mean = self.qw_mean.repeat(self.n_samples, 1)
        qw_var = qw_var.repeat(self.n_samples, 1)
        # assert (qw_mean.shape == w_sample_size)

        w_sample = qw_mean + qw_var * torch.randn(w_sample_size)
        print(f'w-samples: {w_sample.shape}')
        b_sample_size = (self.out_features * self.n_samples, 1)
        b_mean = self.qb_mean.view(-1, 1).repeat(self.n_samples, 1)
        qb_var = qb_var.view(-1, 1).repeat(self.n_samples, 1)

        b_sample = b_mean + qb_var * torch.randn(b_sample_size)
        print(f'b-samples: {b_sample.shape}')
        y = F.linear(input=x, weight=w_sample, )
        print(f'y: {y.shape}')
        log_q_theta = self._log_q_theta(w_sample=w_sample, b_sample=b_sample,
                                        qw_mean=self.qw_mean, qw_std=qw_var,
                                        qb_mean=self.qb_mean, qb_std=qb_var)

        log_p_theta = self._log_p_theta(w_sample=w_sample, b_sample=b_sample)

        # kl_qw_pw = log_p_theta - log_q_theta
        kl_qw_pw = 0
        return y, kl_qw_pw

    def forward(self, x):
        # print()
        # print(f'qw_mean: {self.qw_mean}')
        # print(f'qw_logvar: {self.qw_logvar}')
        # print(f'qb_mean: {self.qb_mean}')
        # print(f'qb_logvar: {self.qb_logvar}')
        # EQUATION
        # y = x·(mu_w + exp(log_var_w + 1.0)·N(0,1)) +
        #     (mu_b + + exp(log_var_b + 1.0)·N(0,1))
        batch_size = x.shape[0]

        qw_var = torch.exp(1.0 + self.qw_logvar)
        qb_var = torch.exp(1.0 + self.qb_logvar)

        x_w_mu = F.linear(input=x, weight=self.qw_mean)
        x_w_var = F.linear(input=x, weight=qw_var)

        b_mu = self.qb_mean.repeat(batch_size, 1)
        b_var = qb_var.repeat(batch_size, 1)

        output_size = x_w_mu.size()
        w_samples = torch.randn(output_size)
        b_samples = torch.randn(output_size)
        if self.is_cuda:
            w_samples = w_samples.cuda()
            b_samples = b_samples.cuda()

        # print()
        # print(mu_b.shape)
        # print(log_var_b.shape)
        # print(b_samples.shape)
        output = 1e-8 + x_w_mu + x_w_var * w_samples + \
                 b_mu + b_var * b_samples

        # calculate KL(q(theta)||p(theta))
        kl_qw_pw = self.get_kl_qw_pw(torch.sqrt(qb_var), torch.sqrt(qw_var))
        if self.is_cuda:
            kl_qw_pw = kl_qw_pw.cuda()
        return output, kl_qw_pw

    def get_kl_qw_pw(self, qb_std, qw_std):
        kl_qw_pw = 0.0
        qw_mean = self.qw_mean
        qb_mean = self.qb_mean
        if self.is_cuda:
            qw_mean = qw_mean.cpu()
            qb_mean = qb_mean.cpu()
            qb_std = qb_std.cpu()
            qw_std = qw_std.cpu()

        qw = Normal(loc=qw_mean, scale=qw_std)
        qb = Normal(loc=qb_mean, scale=qb_std)
        pw = Normal(loc=torch.zeros(qw_mean.shape), scale=torch.ones(qw_mean.shape))
        pb = Normal(loc=torch.zeros(qb_mean.shape), scale=torch.ones(qb_mean.shape))
        # if self.is_cuda:
        #     qw = qw.cuda()
        #     qb = qb.cuda()
        #     pw = pw.cuda()
        #     pb = pb.cuda()

        for _ in range(self.n_samples):
            qw_sample = qw.sample()
            qb_sample = qb.sample()

            log_qw = qw.log_prob(qw_sample)
            log_qb = qb.log_prob(qb_sample)
            log_q_theta = torch.sum(log_qb.reshape((-1))) + torch.sum(log_qw.reshape((-1)))


            log_pw = pw.log_prob(qw_sample)
            log_pb = pb.log_prob(qb_sample)
            log_p_theta = torch.sum(log_pb.reshape((-1))) + torch.sum(log_pw.reshape((-1)))

            kl_qw_pw += log_q_theta - log_p_theta
        kl_qw_pw /= self.n_samples

        # 2nd alternative to calculate kl divergence
        # The Kullback–Leibler divergence is additive for independent distributions
        # kl_w_qw_pw = kl_divergence(qw, pw).sum()
        # kl_b_qw_pw = kl_divergence(qb, pb).sum()
        # kl_qw_pw_2 = kl_b_qw_pw + kl_w_qw_pw
        # print(f'kl_qw_pw on distribution: {kl_qw_pw_2}')
        # print(f'kl_qw_pw on samples: {kl_qw_pw}')
        return kl_qw_pw

    def _compute_log(self, mean, std, samples):
        q_theta_sample = mean + std * samples
        log_qw_sample = torch.log(q_theta_sample)
        return log_qw_sample

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'

    def _log_q_theta(self, w_sample, b_sample, qw_mean, qw_std, qb_mean, qb_std):
        return None

    def _log_p_theta(self, w_sample, b_sample):
        return None


class ComplexStochasticLinear(nn.Module):

    def __init__(self, in_features, out_features, is_cuda=False, parameters: StochasticLinearParameters = None,
                 n_samples=10, q_logvar_init=-5, is_trainable=False):
        # p_logvar_init, p_pi can be either
        # (list/tuples): prior model is a mixture of Gaussians components=len(p_pi)=len(p_logvar_init)
        # float: Gussian distribution
        # q_logvar_init: float, the approximate posterior is currently always a factorized gaussian
        super(ComplexStochasticLinear, self).__init__()
        self.is_trainable = is_trainable

        self.in_features = in_features
        self.out_features = out_features
        self.is_cuda = is_cuda
        self.q_logvar_init = q_logvar_init
        self.n_samples = n_samples

        self.qw_mean = None
        self.qw_logvar = None
        self.qb_mean = None
        self.qb_logvar = None
        self.log_alpha = None

        if parameters is None:
            # Approximate posterior weights and biases
            self.qw_mean = Parameter(torch.Tensor(out_features, in_features))
            self.qw_logvar = Parameter(torch.Tensor(out_features, in_features))

            # optionally add bias
            self.qb_mean = Parameter(torch.Tensor(out_features))
            self.qb_logvar = Parameter(torch.Tensor(out_features))

            self.log_alpha = Parameter(torch.Tensor(1, 1))

            # initialize all paramaters
            self.reset_parameters()
        else:
            # this parameters are known
            # print(f'qw_mean: {parameters.qw_mean.shape}')
            # print(f'qw_logvar: {parameters.qw_logvar.shape}')
            # print(f'qb_mean: {parameters.qb_mean.shape}')
            # print(f'qb_logvar: {parameters.qb_logvar.shape}')
            self.qw_mean = parameters.qw_mean
            self.qw_logvar = parameters.qw_logvar

            self.qb_mean = parameters.qb_mean
            self.qb_logvar = parameters.qb_logvar

            if parameters.log_alpha is not None:
                self.log_alpha = parameters.log_alpha

            if self.is_trainable:
                self.qw_mean = Parameter(self.qw_mean)
                self.qw_logvar = Parameter(self.qw_logvar)

                self.qb_mean = Parameter(self.qb_mean)
                self.qb_logvar = Parameter(self.qb_logvar)
                self.log_alpha = Parameter(self.log_alpha)


    def reset_parameters(self):
        # initialize (trainable) approximate posterior parameters
        stdv = 10. / math.sqrt(self.in_features)
        self.qw_mean.data.uniform_(-stdv, stdv)
        self.qw_logvar.data.uniform_(-stdv, stdv).add_(self.q_logvar_init)
        self.qb_mean.data.uniform_(-stdv, stdv)
        self.qb_logvar.data.uniform_(-stdv, stdv).add_(self.q_logvar_init)

        # assumes 1 sigma for all weights per layer.
        self.log_alpha.data.uniform_(-stdv, stdv)

    def forward_2(self, x):
        # EQUATION
        # y = x·(mu_w + exp(log_var_w + 1.0)·N(0,1)) +
        #     (mu_b + + exp(log_var_b + 1.0)·N(0,1))
        batch_size = x.shape[0]

        print(f'x: {x.shape}')
        qw_var = torch.exp(1.0 + self.qw_logvar)
        qb_var = torch.exp(1.0 + self.qb_logvar)

        w_sample_size = (self.out_features * self.n_samples, self.in_features)
        qw_mean = self.qw_mean.repeat(self.n_samples, 1)
        qw_var = qw_var.repeat(self.n_samples, 1)
        # assert (qw_mean.shape == w_sample_size)

        w_sample = qw_mean + qw_var * torch.randn(w_sample_size)
        print(f'w-samples: {w_sample.shape}')
        b_sample_size = (self.out_features * self.n_samples, 1)
        b_mean = self.qb_mean.view(-1, 1).repeat(self.n_samples, 1)
        qb_var = qb_var.view(-1, 1).repeat(self.n_samples, 1)

        b_sample = b_mean + qb_var * torch.randn(b_sample_size)
        print(f'b-samples: {b_sample.shape}')
        y = F.linear(input=x, weight=w_sample, )
        print(f'y: {y.shape}')
        log_q_theta = self._log_q_theta(w_sample=w_sample, b_sample=b_sample,
                                        qw_mean=self.qw_mean, qw_std=qw_var,
                                        qb_mean=self.qb_mean, qb_std=qb_var)

        log_p_theta = self._log_p_theta(w_sample=w_sample, b_sample=b_sample)

        # kl_qw_pw = log_p_theta - log_q_theta
        kl_qw_pw = 0
        return y, kl_qw_pw

    def forward(self, x):
        # print()
        # print(f'qw_mean: {self.qw_mean}')
        # print(f'qw_logvar: {self.qw_logvar}')
        # print(f'qb_mean: {self.qb_mean}')
        # print(f'qb_logvar: {self.qb_logvar}')
        # EQUATION
        # y = x·(mu_w + exp(log_var_w + 1.0)·N(0,1)) +
        #     (mu_b + + exp(log_var_b + 1.0)·N(0,1))
        batch_size = x.shape[0]

        qw_var = torch.exp(1.0 + self.qw_logvar)
        qb_var = torch.exp(1.0 + self.qb_logvar)

        x_w_mu = F.linear(input=x, weight=self.qw_mean)
        x_w_var = F.linear(input=x, weight=qw_var)

        b_mu = self.qb_mean.repeat(batch_size, 1)
        b_var = qb_var.repeat(batch_size, 1)

        output_size = x_w_mu.size()
        w_samples = torch.randn(output_size)
        b_samples = torch.randn(output_size)
        if self.is_cuda:
            w_samples = w_samples.cuda()
            b_samples = b_samples.cuda()

        # print()
        # print(mu_b.shape)
        # print(log_var_b.shape)
        # print(b_samples.shape)
        output = 1e-8 + x_w_mu + x_w_var * w_samples + \
                 b_mu + b_var * b_samples

        return output

    def get_kl_qw_pw(self, qb_std, qw_std):
        kl_qw_pw = 0.0
        qw_mean = self.qw_mean
        qb_mean = self.qb_mean
        if self.is_cuda:
            qw_mean = qw_mean.cpu()
            qb_mean = qb_mean.cpu()
            qb_std = qb_std.cpu()
            qw_std = qw_std.cpu()

        qw = Normal(loc=qw_mean, scale=qw_std)
        qb = Normal(loc=qb_mean, scale=qb_std)
        pw = Normal(loc=torch.zeros(qw_mean.shape), scale=torch.ones(qw_mean.shape))
        pb = Normal(loc=torch.zeros(qb_mean.shape), scale=torch.ones(qb_mean.shape))
        # if self.is_cuda:
        #     qw = qw.cuda()
        #     qb = qb.cuda()
        #     pw = pw.cuda()
        #     pb = pb.cuda()

        for _ in range(self.n_samples):
            qw_sample = qw.sample()
            qb_sample = qb.sample()

            log_qw = qw.log_prob(qw_sample)
            log_qb = qb.log_prob(qb_sample)
            log_q_theta = torch.sum(log_qb.reshape((-1))) + torch.sum(log_qw.reshape((-1)))


            log_pw = pw.log_prob(qw_sample)
            log_pb = pb.log_prob(qb_sample)
            log_p_theta = torch.sum(log_pb.reshape((-1))) + torch.sum(log_pw.reshape((-1)))

            kl_qw_pw += log_q_theta - log_p_theta
        kl_qw_pw /= self.n_samples

        # 2nd alternative to calculate kl divergence
        # The Kullback–Leibler divergence is additive for independent distributions
        # kl_w_qw_pw = kl_divergence(qw, pw).sum()
        # kl_b_qw_pw = kl_divergence(qb, pb).sum()
        # kl_qw_pw_2 = kl_b_qw_pw + kl_w_qw_pw
        # print(f'kl_qw_pw on distribution: {kl_qw_pw_2}')
        # print(f'kl_qw_pw on samples: {kl_qw_pw}')
        return kl_qw_pw

    def _compute_log(self, mean, std, samples):
        q_theta_sample = mean + std * samples
        log_qw_sample = torch.log(q_theta_sample)
        return log_qw_sample

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'

    def _log_q_theta(self, w_sample, b_sample, qw_mean, qw_std, qb_mean, qb_std):
        return None

    def _log_p_theta(self, w_sample, b_sample):
        return None