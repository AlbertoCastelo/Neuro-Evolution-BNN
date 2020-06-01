import numpy as np
import pandas as pd


def expected_calibration_error(y_true, y_pred_prob, n_bins):
    # n_bins_width = 1 / n_bins
    bins = [i / n_bins for i in range(1, n_bins + 1)]
    # bins_confidence = [i / n_bins + n_bins_width / 2 for i in range(n_bins)]

    classes = np.unique(y_true)

    calibration_data_chunks = []
    ece = 0.0
    for class_ in classes:
        class_weight = np.mean(np.where(y_true == class_, 1, 0))
        ece_class, calibration_data_class = confidence_calibration(bins, class_, n_bins, y_pred_prob, y_true)
        ece += ece_class * class_weight

        calibration_data_class['class'] = class_
        calibration_data_chunks.append(calibration_data_class)
    calibration_data = pd.concat(calibration_data_chunks)
    return ece, calibration_data


def confidence_calibration(bins, class_, n_bins, y_pred_prob, y_true):
    y_true_class = np.where(y_true == class_, 1, 0)
    y_pred_prob_class = y_pred_prob[:, class_]
    # fraction_of_positives, mean_predicted_value = calibration_curve(y_true=y_true_class, y_prob=y_pred_prob_class,
    #                                                                 n_bins=n_bins)
    # ece_class = abs(fraction_of_positives - mean_predicted_value)
    # # indices = np.where(y_true == class_)[0]
    # # probs = np.take(y_pred_prob, indices)
    probs = y_pred_prob[:, class_]
    bin_index = np.digitize(probs, bins)
    chunks = []
    for bin in range(n_bins):
        # probs_bin = np.take(y_true_class, indices)
        indices_prob = np.where(bin_index == bin)[0]
        y_true_class_bin = np.take(y_true_class, indices_prob)
        y_pred_prob_class_bin = np.take(y_pred_prob_class, indices_prob)
        n_values_bin = len(y_true_class_bin)
        acc_bin = np.mean(y_true_class_bin)
        confidence_bin = np.mean(y_pred_prob_class_bin)

        abs_error = np.abs(acc_bin - confidence_bin)
        chunks.append([bins[bin], n_values_bin, acc_bin, confidence_bin, abs_error])
    calibration_data = pd.DataFrame(chunks, columns=['bin_high_limit', 'n_values_bin', 'acc_bin',
                                                     'confidence_bin', 'abs_error'])
    calibration_data['weighted_abs_error'] = calibration_data['n_values_bin'] * calibration_data['abs_error']
    ece = calibration_data['weighted_abs_error'].sum() / calibration_data['n_values_bin'].sum()
    return ece, calibration_data
