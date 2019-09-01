import pandas as pd
import numpy as np

from deep_learning.probabilistic.train_eval import EvaluateProbabilisticDL
from neat.dataset.classification_example import ClassificationExample1Dataset
import seaborn as sns
from tests.config_files.config_files import create_configuration
import matplotlib.pyplot as plt
from deep_learning.standard.train_eval import EvaluateStandardDL


config_file = '/classification-miso.json'
network_filename = f'network-probabilistic-classification.pt'
dataset = ClassificationExample1Dataset()

config = create_configuration(filename=config_file)
is_cuda = False

n_samples = 100
lr = 0.01
weight_decay = 0.0005
n_epochs = 2

batch_size = 50000

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

# evaluator.save_network(network_filename)

# predict
print('Evaluating results')
x, y_true, y_pred = evaluator.evaluate()

x = dataset.input_scaler.inverse_transform(x)
# y_true =y_true.numpy()


# plot results

y_pred = np.argmax(y_pred.numpy(), 1)
df = pd.DataFrame(x, columns=['x1', 'x2'])
df['y'] = y_pred

x1_limit, x2_limit = dataset.get_separation_line()

plt.figure()
ax = sns.scatterplot(x='x1', y='x2', hue='y', data=df)
ax.plot(x1_limit, x2_limit, 'g-', linewidth=2.5)
plt.show()

from sklearn.metrics import confusion_matrix, accuracy_score
print('Confusion Matrix:')
print(confusion_matrix(y_true, y_pred))

print(f'Accuracy: {accuracy_score(y_true, y_pred)*100} %')
