import pandas as pd
import numpy as np
import torch

from config_files.configuration_utils import create_configuration
from deep_learning.probabilistic.evaluate_probabilistic_dl import EvaluateProbabilisticDL
import seaborn as sns
import matplotlib.pyplot as plt

from neat.evaluation.utils import get_dataset

# dataset_name = 'classification-miso'
# dataset_name = 'iris'
dataset_name = 'mnist_downsampled'

config = create_configuration(filename=f'/{dataset_name}.json')
config.noise = 0.0
# config.n_input = 64
config.n_output = 10

config.n_samples = 100
# network_filename = f'network-probabilistic-classification.pt'
dataset = get_dataset(dataset=config.dataset,
                      train_percentage=config.train_percentage,
                      random_state=config.dataset_random_state,
                      noise=config.noise)

is_cuda = True

lr = 0.01
weight_decay = 0.0005
n_epochs = 5000

batch_size = 50000

if is_cuda:
    use_cuda = torch.cuda.is_available()
    torch.cuda.set_device(0)

evaluator = EvaluateProbabilisticDL(dataset=dataset,
                                    batch_size=batch_size,
                                    n_samples=config.n_samples,
                                    lr=lr,
                                    weight_decay=weight_decay,
                                    n_epochs=n_epochs,
                                    n_neurons_per_layer=10,
                                    n_hidden_layers=1,
                                    is_cuda=is_cuda,
                                    beta=0.0)
evaluator.run()

# evaluator.save_network(network_filename)

# predict
print('Evaluating results')
x, y_true, y_pred = evaluator.evaluate()

if is_cuda:
    x = x.cpu()
    y_true = y_true.cpu()
    y_pred = y_pred.cpu()
# x = dataset.input_scaler.inverse_transform(x)
y_true = y_true.numpy()
y_pred = torch.argmax(y_pred, dim=1).numpy()

# plot results
#
# y_pred = np.argmax(y_pred.numpy(), 1)
# df = pd.DataFrame(x, columns=['x1', 'x2'])
# df['y'] = y_pred
#
# x1_limit, x2_limit = dataset.get_separation_line()
#
# plt.figure()
# ax = sns.scatterplot(x='x1', y='x2', hue='y', data=df)
# ax.plot(x1_limit, x2_limit, 'g-', linewidth=2.5)
# plt.show()

from sklearn.metrics import confusion_matrix, accuracy_score
print('Confusion Matrix:')
print(confusion_matrix(y_true, y_pred))

print(f'Accuracy: {accuracy_score(y_true, y_pred)*100} %')
