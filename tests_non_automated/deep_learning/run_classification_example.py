import torch
import pandas as pd
import numpy as np
from neat.dataset.classification_example import ClassificationExample1Dataset
from neat.dataset.regression_example import RegressionExample1Dataset
import seaborn as sns
from tests.config_files.config_files import create_configuration
import matplotlib.pyplot as plt
from tests_non_automated.deep_learning.train_eval import EvaluateStandardDL


config_file = '/classification-miso.json'
network_filename = f'network-classification.pt'
dataset = ClassificationExample1Dataset()

config = create_configuration(filename=config_file)
is_cuda = False

lr = 0.01
weight_decay = 0.0005
n_epochs = 30


batch_size = 50000

evaluator = EvaluateStandardDL(dataset=dataset,
                               batch_size=batch_size,
                               lr=lr,
                               weight_decay=weight_decay,
                               n_epochs=n_epochs,
                               n_neurons_per_layer=5,
                               n_hidden_layers=2,
                               is_cuda=is_cuda)
evaluator.run()

evaluator.save_network(network_filename)

# predict
x_test = dataset.x
y_true = dataset.y
y_pred = evaluator.predict(x_test)

x_test = dataset.input_scaler.inverse_transform(x_test)
# y_pred = dataset.output_transformer.inverse_transform(y_pred)

# plot results
df = pd.DataFrame(x_test, columns=['x1', 'x2'])
df['y'] = np.argmax(y_pred.numpy(), 1)

x1_limit, x2_limit = dataset.get_separation_line()

plt.figure()
ax = sns.scatterplot(x='x1', y='x2', hue='y', data=df)
ax.plot(x1_limit, x2_limit, 'g-', linewidth=2.5)
plt.show()

