import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def plot_loss_distribution(data):
    loss_data = pd.melt(data, id_vars=['correlation_id', 'execution_id', 'is_bayesian', 'train_percentage'],
                        value_vars=['loss_training', 'loss_testing'], var_name='type', value_name='loss')

    correlation_ids = loss_data['correlation_id'].unique().tolist()
    _, axes = plt.subplots(len(correlation_ids), 1, figsize=(15, 8))
    for i, correlation_id in enumerate(correlation_ids):
        if len(correlation_ids) > 1:
            ax = axes[i]
        else:
            ax = axes
        sns.boxplot(data=loss_data.loc[loss_data['correlation_id'] == correlation_id],
                    x='train_percentage', y='loss', hue='type', ax=ax)
        ax.set_title(f'Experiment: {correlation_id}', size=20)
    plt.show()

    print()
    print()
    print()
    print('COMPARE LOSSES IN NEAT VS. BAYESIAN-NEAT')
    plt.figure(figsize=(15, 8))
    sns.boxplot(data=loss_data, x='is_bayesian', y='loss', hue='type')
