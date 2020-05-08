import numpy as np
import matplotlib.pyplot as plt

from neat.analysis.plotting.plot_pca import plot_dimensionality_reduction
from neat.analysis.uncertainty.predictive_distribution import PredictionDistributionEstimator


def plot_prediction_probabilities(estimator: PredictionDistributionEstimator, index_to_plot, plot_pca=True):
    n_classes = estimator.output_distribution.shape[-1]
    nbins = 50
    output_distribution_example = estimator.output_distribution.numpy()[index_to_plot]
    y_true_example = estimator.y_true.numpy()[index_to_plot]
    y_pred_example = estimator.y_pred.numpy()[index_to_plot]

    std = estimator.output_stds.numpy()[index_to_plot]

    print(f'True -> {y_true_example}')
    print(f'Predicted -> {y_pred_example}')
    print(f'STDS: {std}')

    n_rows = n_classes
    if plot_pca:
        n_rows += 1

    _, axes = plt.subplots(1, n_rows, figsize=(20, 8))
    for label in range(n_classes):
        ax = axes[label]
        ax.set_title(f'P(y={label}| x)', size=20)
        if estimator.is_bayesian:
            # show histogram
            array_label = np.round(output_distribution_example[:, label], 5)
            ax.hist(array_label, bins=_bins_generator(array=array_label, nbins=nbins))
            ax.axvline(np.mean(array_label), color='r')
        else:
            # show barplot
            p_label = np.round(output_distribution_example[:, label].mean(), 5)
            ax.bar(p_label, 1, width=0.01)

            ax.set_xlim(_get_limits(p_label, width_axes=0.15))

    if plot_pca:
        dataset = estimator.get_dataset()
        if estimator.testing:
            x = dataset.x_test.numpy()
            y = dataset.y_test.numpy()
        else:
            x = dataset.x_train.numpy()
            y = dataset.y_train.numpy()
        plot_dimensionality_reduction(x, y, index_to_plot, ax=axes[-1])

    plt.show()


def _get_limits(p_label, width_axes=0.15):
    half_width = width_axes / 2

    if p_label < 0.5:
        low = p_label - half_width
        sum_high = 0.0
        if low < 0.0:
            sum_high = - low
            low = 0.0

        high = p_label + half_width + sum_high
    else:
        high = p_label + half_width
        sum_low = 0.0
        if high > 1.0:
            sum_low = (high - 1)
            high = 1.0

        low = p_label - half_width - sum_low
    return low, high


def _bins_generator(array, nbins):
    min_ = np.min(array)
    max_ = np.max(array)
    bin_size = (max_ - min_) / nbins

    acc = min_
    bins = [acc]
    for i in range(nbins):
        acc += bin_size
        bins.append(acc)
    return bins
