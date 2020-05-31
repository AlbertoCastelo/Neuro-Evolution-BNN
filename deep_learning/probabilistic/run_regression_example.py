import torch

from config_files.configuration_utils import create_configuration
from deep_learning.probabilistic.evaluate_probabilistic_dl import EvaluateProbabilisticDL
from neat.dataset.regression_example import RegressionExample1Dataset

import matplotlib.pyplot as plt

from neat.evaluation.utils import get_dataset

# config_file = '/regression-siso.json'
dataset_name = 'regression-siso'
config = create_configuration(filename=f'/{dataset_name}.json')
config.train_percentage = 0.75
config.n_samples = 100
# network_filename = f'network-probabilistic-classification.pt'
dataset = get_dataset(dataset=config.dataset,
                      train_percentage=config.train_percentage,
                      random_state=config.dataset_random_state,
                      noise=config.noise,
                      label_noise=config.label_noise)

# TODO: fix Memory-leakage in this network when doing backprop
n_samples = 1000
is_cuda = False

lr = 0.01
weight_decay = 0.0005
n_epochs = 1000

batch_size = 50000

if is_cuda:
    use_cuda = torch.cuda.is_available()
    torch.cuda.set_device(0)

# torch.set_num_threads(1)
evaluator = EvaluateProbabilisticDL(dataset=dataset,
                                    batch_size=batch_size,
                                    n_samples=n_samples,
                                    lr=lr,
                                    weight_decay=weight_decay,
                                    n_epochs=n_epochs,
                                    n_neurons_per_layer=10,
                                    n_hidden_layers=1,
                                    is_cuda=is_cuda,
                                    beta=0.0001)
evaluator.run()

# evaluator.save_network(network_filename)

# predict
print('Evaluating')
x, y_true, y_pred = evaluator.evaluate()
if is_cuda:
    x = x.cpu()
    y_true = y_true.cpu()
    y_pred = y_pred.cpu()

print(x.shape)
x = dataset.input_scaler.inverse_transform(x.numpy())
y_pred = dataset.output_scaler.inverse_transform(y_pred.numpy())
y_true = dataset.output_scaler.inverse_transform(y_true.numpy())
# y_pred = evaluator.predict(x_test).numpy()
#
plt.figure(figsize=(20, 20))
plt.plot(x, y_true, 'r*')
plt.plot(x, y_pred, 'b*')
plt.show()
