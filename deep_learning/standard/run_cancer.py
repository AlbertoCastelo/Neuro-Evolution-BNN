import pandas as pd
import numpy as np
import os
from config_files.configuration_utils import create_configuration
from neat.dataset.classification_example import ClassificationExample1Dataset
import seaborn as sns
import matplotlib.pyplot as plt
from deep_learning.standard.train_eval import EvaluateStandardDL

# DATASET = 'mnist'
from neat.evaluation.utils import get_dataset
from neat.neat_logger import get_neat_logger

DATASET = 'cancer'

config = create_configuration(filename=f'/{DATASET}.json')
config.n_input = 32*32
# config.n_output = 4
LOGS_PATH = f'{os.getcwd()}/'
logger = get_neat_logger(path=LOGS_PATH)

network_filename = f'network-{DATASET}.pt'
dataset = get_dataset(dataset=config.dataset, train_percentage=0.5)

is_cuda = False

lr = 0.01
weight_decay = 0.0005
n_epochs = 1500
batch_size = 50000

evaluator = EvaluateStandardDL(dataset=dataset,
                               batch_size=batch_size,
                               lr=lr,
                               weight_decay=weight_decay,
                               n_epochs=n_epochs,
                               n_neurons_per_layer=10,
                               n_hidden_layers=1,
                               is_cuda=is_cuda)
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


from sklearn.metrics import confusion_matrix, accuracy_score
print('Confusion Matrix:')
print(confusion_matrix(y_true, y_pred))

print(f'Accuracy: {accuracy_score(y_true, y_pred)*100} %')
