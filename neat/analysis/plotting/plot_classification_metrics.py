import matplotlib.pyplot as plt


def plot_metrics_by_quantile(metrics_by_quantile):

    _, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    ax1.plot(metrics_by_quantile['order_std'], metrics_by_quantile['accuracy'])
    ax1.set_title('Accuracy')

    ax2.plot(metrics_by_quantile['order_std'], metrics_by_quantile['f1'])
    ax2.set_title('F1 Score')


def plot_metrics_by_quantile_several_executions(metrics_by_dispersion_quantile):
    _, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    for execution_id, metrics_by_execution_id in metrics_by_dispersion_quantile.groupby('execution_id'):
        ax1.plot(metrics_by_execution_id['order_std'], metrics_by_execution_id['accuracy'])
        ax1.set_title('Accuracy')
        # ax1.set_x

        ax2.plot(metrics_by_execution_id['order_std'], metrics_by_execution_id['f1'])
        ax2.set_title('F1 Score')
    plt.show()
