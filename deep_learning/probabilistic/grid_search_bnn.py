from unittest.mock import Mock

import torch
import os
from config_files.configuration_utils import create_configuration
from deep_learning.probabilistic.nas import neural_architecture_search
from experiments.reporting.report_repository import ReportRepository
from experiments.slack_client import SlackNotifier
from neat.neat_logger import get_neat_logger

dataset_name = 'iris'
# dataset_name = 'mnist_downsampled'
# dataset_name = 'titanic'
CORRELATION_ID = 'nas_v2'
# CORRELATION_ID = 'nas_v1'
N_REPETITIONS = 2
is_debug = False

## PARAMETERS THAT WON'T CHANGE MUCH
N_HIDDEN_LAYERS_VALUES = [1, 2]
N_NEURONS_PER_LAYER_VALUES = list(range(2, 21))
# N_HIDDEN_LAYERS_VALUES = [1]
# N_NEURONS_PER_LAYER_VALUES = [10]
NOISES = [0.0]
# NOISES = [0.0, 0.5, 1.0, 2.0, 5.0]

LOGS_PATH = f'{os.getcwd()}/'
logger = get_neat_logger(path=LOGS_PATH)

is_cuda = False
lr = 0.01
weight_decay = 0.0005
n_epochs = 2000
batch_size = 50000

report_repository = ReportRepository.create(project='nas', logs_path=LOGS_PATH)
notifier = SlackNotifier.create(channel='batch-jobs')

config = create_configuration(filename=f'/{dataset_name}.json')
config.noise = 0.0
config.train_percentage = 0.75
config.n_samples = 100

if is_cuda:
    use_cuda = torch.cuda.is_available()
    torch.cuda.set_device(0)

if is_debug:
    N_HIDDEN_LAYERS_VALUES = [1]
    N_NEURONS_PER_LAYER_VALUES = list(range(2, 3))
    n_epochs = 1
    notifier = Mock()
    CORRELATION_ID += 'debugging'
    N_REPETITIONS = 1


for rep in range(N_REPETITIONS):
    for noise in NOISES:
        config.noise = noise
        neural_architecture_search(n_hidden_layers_values=N_HIDDEN_LAYERS_VALUES,
                                   n_neurons_per_layer_values=N_NEURONS_PER_LAYER_VALUES,
                                   correlation_id=CORRELATION_ID,
                                   config=config,
                                   batch_size=batch_size,
                                   lr=lr,
                                   weight_decay=weight_decay,
                                   n_epochs=n_epochs,
                                   notifier=notifier,
                                   report_repository=report_repository,
                                   is_cuda=False)
