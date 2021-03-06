import random
from unittest.mock import Mock

import torch
import os
from config_files.configuration_utils import create_configuration
from deep_learning.nas import neural_architecture_search
from deep_learning.probabilistic.evaluate_probabilistic_dl import EvaluateProbabilisticDL
from experiments.reporting.report_repository import ReportRepository
from experiments.slack_client import SlackNotifier
from neat.neat_logger import get_neat_logger

# dataset_name = 'iris'
from neat.utils import get_slack_channel

dataset_name = 'mnist_downsampled'
# dataset_name = 'titanic'
# dataset_name = 'classification-miso'
# dataset_name = 'breast_cancer'
# dataset_name = 'spambase'

CORRELATION_ID = 'bayesian_nas_final_v2'     # using update Stochastic layer


N_REPETITIONS = 5
is_debug = False

## PARAMETERS THAT WON'T CHANGE MUCH
N_HIDDEN_LAYERS_VALUES = [1, 2]
N_NEURONS_PER_LAYER_VALUES = [5, 10, 15, 20]

LABEL_NOISES = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]


LOGS_PATH = f'{os.getcwd()}/'
logger = get_neat_logger(path=LOGS_PATH)

is_cuda = True
lr = 0.01
weight_decay = 0.0005
n_epochs = 2000
batch_size = 50000

report_repository = ReportRepository.create(project='nas', logs_path=LOGS_PATH)
notifier = SlackNotifier.create(channel=get_slack_channel(dataset_name=dataset_name))

config = create_configuration(filename=f'/{dataset_name}.json')
config.noise = 0.0
config.train_percentage = 0.75
config.n_samples = 50

if is_cuda:
    use_cuda = torch.cuda.is_available()
    torch.cuda.set_device(0)
    print('Using GPU')

if is_debug:
    N_HIDDEN_LAYERS_VALUES = [1]
    N_NEURONS_PER_LAYER_VALUES = list(range(2, 3))
    n_epochs = 1
    notifier = Mock()
    CORRELATION_ID += 'debugging'
    N_REPETITIONS = 1


for i in range(N_REPETITIONS):
    for label_noise in LABEL_NOISES:
        config.label_noise = label_noise

        config.dataset_random_state = random.sample(list(range(100)), k=1)[0]
        neural_architecture_search(EvaluateDL=EvaluateProbabilisticDL,
                                   n_hidden_layers_values=N_HIDDEN_LAYERS_VALUES,
                                   n_neurons_per_layer_values=N_NEURONS_PER_LAYER_VALUES,
                                   correlation_id=CORRELATION_ID,
                                   config=config,
                                   batch_size=batch_size,
                                   lr=lr,
                                   weight_decay=weight_decay,
                                   n_epochs=n_epochs,
                                   notifier=notifier,
                                   report_repository=report_repository,
                                   is_cuda=is_cuda,
                                   n_repetitions=2)
