from unittest import TestCase, skip
import numpy as np
from neat.analysis.uncertainty.calibration_error import expected_calibration_error
import tests.fixtures.ece_fixtures as fixtures


class TestCalibrationError(TestCase):
    def test_expected_calibration_error(self):
        y_true = np.array(fixtures.y_true)

        y_pred_prob = np.array(fixtures.y_pred_prob)
        ece, calibration_data = expected_calibration_error(y_true=y_true, y_pred_prob=y_pred_prob, n_bins=2)

        self.assertAlmostEqual(ece, 0.03028, places=2)

    def test_expected_calibration_error_by_percentiles(self):
        y_true = np.array(fixtures.y_true)

        y_pred_prob = np.array(fixtures.y_pred_prob)
        ece, calibration_data = expected_calibration_error(y_true=y_true,
                                                           y_pred_prob=y_pred_prob,
                                                           n_bins=2,
                                                           uniform_binning=False)

        self.assertAlmostEqual(ece, 0.0360, places=2)
