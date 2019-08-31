import torch

from neat.dataset.regression_example import RegressionExample1Dataset

from tests.config_files.config_files import create_configuration
import matplotlib.pyplot as plt
from deep_learning.train_eval import EvaluateStandardDL

config_file = '/siso.json'
dataset = RegressionExample1Dataset()
network_filename = f'network-regression_1.pt'


config = create_configuration(filename=config_file)

lr = 0.01
weight_decay = 0.0005
n_epochs = 500

batch_size = 50000

evaluator = EvaluateStandardDL(dataset=dataset,
                               batch_size=batch_size,
                               lr=lr,
                               weight_decay=weight_decay,
                               n_epochs=n_epochs,
                               n_neurons_per_layer=10,
                               n_hidden_layers=1,
                               is_cuda=False)
evaluator.run()

evaluator.save_network(network_filename)

# predict
x_test = torch.Tensor(dataset.x)
y_true = dataset.y
y_pred = evaluator.predict(x_test).numpy()

plt.figure(figsize=(20, 20))
plt.plot(x_test.numpy(), y_true, 'r*')
plt.plot(x_test.numpy(), y_pred, 'b*')
plt.show()
