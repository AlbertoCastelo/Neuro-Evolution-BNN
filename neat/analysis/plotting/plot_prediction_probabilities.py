import numpy as np
import matplotlib.pyplot as plt
import torch


def plot_prediction_probabilities(output_distribution: torch.Tensor,
                                  y_true: torch.Tensor,
                                  y_pred: torch.Tensor,
                                  output_means, output_stds, index_to_plot):
    n_classes = output_distribution.shape[-1]
    nbins = 50
    output_distribution_example = output_distribution[index_to_plot]
    y_true_example = y_true.numpy()[index_to_plot]
    y_pred_example = y_pred.numpy()[index_to_plot]

    std = output_stds.numpy()[index_to_plot]

    print(f'True -> {y_true_example}')
    print(f'Predicted -> {y_pred_example}')
    print(f'STDS: {std}')

    _, axes = plt.subplots(1, 2, figsize=(15, 8))
    for label in range(n_classes):
        ax = axes[label]
        ax.set_title(f'Probability Distribution of label {label}', size=20)
        array_label = np.round(output_distribution_example[:, label].numpy(), 5)
        ax.hist(array_label, bins=_bins_generator(array=array_label, nbins=nbins))

    plt.show()


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