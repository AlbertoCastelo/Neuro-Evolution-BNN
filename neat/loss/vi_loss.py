from torch import nn
from neat.configuration import get_configuration, ConfigError
from neat.neat_logger import logger


def get_loss(problem_type):
    loss = _get_loss_by_problem(problem_type)
    return VariationalInferenceLoss(loss)


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
        logger.debug(f'Log-Likelihood: {logpy}')
        logger.debug(f'KL_QW_PW: {kl_qw_pw}')

        ll = logpy - beta * kl_qw_pw  # ELBO
        logger.debug(f'ELBO: {ll}')
        loss = -ll
        return loss


def get_beta(beta_type, m, batch_idx, epoch, n_epochs):

    if beta_type == 'Blundell':
        beta = 2 ** (m - (batch_idx + 1)) / (2 ** m - 1)
    elif beta_type == 'Soenderby':
        beta = min(epoch / (n_epochs // 4), 1)
    elif beta_type == 'Standard':
        beta = 1 / m
    else:
        beta = get_configuration().beta

    return beta
