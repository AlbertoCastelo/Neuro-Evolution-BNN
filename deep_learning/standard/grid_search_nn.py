import random
from unittest.mock import Mock

import torch
import os
from config_files.configuration_utils import create_configuration
from deep_learning.nas import neural_architecture_search
from deep_learning.standard.evaluate_standard_dl import EvaluateStandardDL
from experiments.reporting.report_repository import ReportRepository
from experiments.slack_client import SlackNotifier
from neat.neat_logger import get_neat_logger
from neat.utils import get_slack_channel

dataset_name = 'iris'
# dataset_name = 'mnist_downsampled'
# dataset_name = 'titanic'
# dataset_name = 'classification-miso'
# dataset_name = 'breast_cancer'

# CORRELATION_ID = 'standard_nas_v1'
# CORRELATION_ID = 'standard_nas_v2' # using Xabier initiliazation
# CORRELATION_ID = 'standard_nas_v3' # 2 per network and 5 per nas
# CORRELATION_ID = 'standard_nas_v4' # 5 per network and 5 per nas LABEL_NOISES = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
# CORRELATION_ID = 'standard_nas_v5' # 1 rep per network and 20 per nas and each execution has random dataset
CORRELATION_ID = 'standard_nas_final'
CORRELATION_ID = 'standard_nas_final_with_reps'

# CORRELATION_ID = 'standard_nas_v6' # 1 rep per network and 20 per nas and each execution has random dataset with attribute noise

# CORRELATION_ID = 'nas_v1'
N_REPETITIONS = 10
is_debug = False

## PARAMETERS THAT WON'T CHANGE MUCH
# N_HIDDEN_LAYERS_VALUES = [3]
# N_NEURONS_PER_LAYER_VALUES = [20]
N_HIDDEN_LAYERS_VALUES = [1, 2]
N_NEURONS_PER_LAYER_VALUES = [5, 10, 15, 20]

# NOISES = [0.0, 1.0, 2.0]


# LABEL_NOISES = [0.6, 0.7, 0.8]
LABEL_NOISES = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
LABEL_NOISES = [0.0]



# NOISES = [0.0, 0.5, 1.0, 2.0, 5.0]

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
# config.label_noise = 0.75
config.train_percentage = 0.75
# config.n_input = 64

# LABEL_NOISES = [i * config.n_output for i in [0.0, 0.25, ] ]

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

    # for noise in NOISES:
    #     config.noise = noise

        config.dataset_random_state = random.sample(list(range(100)), k=1)[0]
        neural_architecture_search(EvaluateDL=EvaluateStandardDL,
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
                                   n_repetitions=10)
