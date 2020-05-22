import pandas as pd
import numpy as np
import os
from config_files.configuration_utils import create_configuration
from neat.dataset.classification_example import ClassificationExample1Dataset
import seaborn as sns
import matplotlib.pyplot as plt
from deep_learning.standard.evaluate_standard_dl import EvaluateStandardDL
from neat.evaluation.utils import get_dataset
from neat.neat_logger import get_neat_logger

DATASET = 'iris'
# DATASET = 'titanic'
# DATASET = 'mnist_downsampled'


config = create_configuration(filename=f'/{DATASET}.json')
config.noise = 0.0
config.label_noise = 0.0
config.train_percentage = 0.75

lr = 0.01
weight_decay = 0.0005
n_epochs = 4000

# config.n_output = 3
LOGS_PATH = f'{os.getcwd()}/'
logger = get_neat_logger(path=LOGS_PATH)

network_filename = f'network-{DATASET}.pt'
dataset = get_dataset(dataset=config.dataset, train_percentage=config.train_percentage,
                      random_state=config.dataset_random_state, noise=config.noise,
                      label_noise=config.label_noise)

is_cuda = False

batch_size = 50000

evaluator = EvaluateStandardDL(dataset=dataset,
                               batch_size=batch_size,
                               lr=lr,
                               weight_decay=weight_decay,
                               n_epochs=n_epochs,
                               n_neurons_per_layer=10,
                               n_hidden_layers=1,
                               is_cuda=is_cuda,
                               n_repetitions=5)
evaluator.run()

evaluator.save_network(network_filename)

# predict
# x_test = dataset.x
# y_true = dataset.y
x, y_true, y_pred = evaluator.evaluate()

x = x.numpy()
y_true = y_true.numpy()
y_pred = y_pred.numpy()

# plot results
y_pred = np.argmax(y_pred, 1)

print('Evaluate on Validation Test')
from sklearn.metrics import confusion_matrix, accuracy_score
print('Confusion Matrix:')
print(confusion_matrix(y_true, y_pred))

print(f'Accuracy: {accuracy_score(y_true, y_pred)*100} %')
