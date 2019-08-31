import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from neat.dataset.classification_example import ClassificationExample1Dataset

dataset = ClassificationExample1Dataset()
dataset.generate_data()

x = dataset.x.numpy()
y = dataset.y.numpy()

x = dataset.input_scaler.inverse_transform(x)
y = dataset.output_transformer.inverse_transform(y)

df = pd.DataFrame(x, columns=['x1', 'x2'])
df['y'] = y

x1_limit, x2_limit = dataset.get_separation_line()

plt.figure()
ax = sns.scatterplot(x='x1', y='x2', hue='y', data=df)
ax.plot(x1_limit, x2_limit, 'g-', linewidth=2.5)
plt.show()
