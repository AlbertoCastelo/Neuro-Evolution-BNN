import torch

from deep_learning.probabilistic.train_eval import EvaluateProbabilisticDL
from neat.dataset.regression_example import RegressionExample1Dataset

from tests.config_files.config_files import create_configuration
import matplotlib.pyplot as plt
from deep_learning.standard.train_eval import EvaluateStandardDL

config_file = '/siso.json'
dataset = RegressionExample1Dataset()
network_filename = f'network-regression_1.pt'


config = create_configuration(filename=config_file)

n_samples = 50
is_cuda = False

lr = 0.01
weight_decay = 0.0005
n_epochs = 2

batch_size = 5000
# torch.set_num_threads(1)
evaluator = EvaluateProbabilisticDL(dataset=dataset,
                                    batch_size=batch_size,
                                    n_samples=n_samples,
                                    lr=lr,
                                    weight_decay=weight_decay,
                                    n_epochs=n_epochs,
                                    n_neurons_per_layer=10,
                                    n_hidden_layers=1,
                                    is_cuda=is_cuda)
evaluator.run()

evaluator.save_network(network_filename)

# predict
print('Evaluating')
x, y_true, y_pred = evaluator.evaluate()
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
