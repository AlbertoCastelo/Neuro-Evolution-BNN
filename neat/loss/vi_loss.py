from torch import nn
from neat.configuration import get_configuration, ConfigError


def get_loss(problem_type):
    loss = _get_loss_by_problem(problem_type)
    return VariationalInferenceLoss(loss)


def get_loss_alternative(problem_type):
    loss = _get_loss_by_problem(problem_type)
    return VariationalInferenceLossAlternative(loss)


def _get_loss_by_problem(problem_type):
    '''
    Regression problems assume Gaussian Likelihood
    Classification problems assume
    '''
    if problem_type == 'regression':
        loss = nn.MSELoss()
    elif problem_type == 'classification':
        loss = nn.CrossEntropyLoss()
    else:
        raise ConfigError(f'Problem Type is incorrect: {problem_type}')
    return loss


class VariationalInferenceLoss(nn.Module):
    '''
    Beta is used to scaled down the KL term when doing batch evaluation
    '''
    def __init__(self, loss=nn.CrossEntropyLoss()):
        super(VariationalInferenceLoss, self).__init__()
        self.loss = loss

    def forward(self, y_pred, y_true, kl_qw_pw, beta):
        # log likelihood
        logpy = -self.loss(y_pred, y_true)

        ll = logpy - beta * kl_qw_pw  # ELBO
        loss = -ll

        return loss


class VariationalInferenceLossAlternative(nn.Module):
    '''
    In this case we don't need to scale down the KL(q(w)||p(w)) because the log p(y) is already through all data
    '''
    def __init__(self, loss=nn.CrossEntropyLoss()):
        super(VariationalInferenceLossAlternative, self).__init__()
        self.loss = loss

    def forward(self, y_pred, y_true):
        # log likelihood
        logpy = -self.loss(y_pred, y_true)
        return logpy

    def compute_complete_loss(self, logpy, kl_qw_pw):
        ll = logpy - kl_qw_pw  # ELBO
        loss = -ll
        return loss