from unittest import TestCase, skip
import numpy as np
from neat.analysis.uncertainty.calibration_error import expected_calibration_error


class TestCalibrationError(TestCase):
    def test_expected_calibration_error(self):
        y_true = np.array([0, 1, 2, 0, 0, 1, 2])
        y_pred_prob = np.array([[0.95, 0.01, 0.04],
                                [0.05, 0.9, 0.05],
                                [0.4, 0.2, 0.4],
                                [0.36, 0.3, 0.34],
                                [0.95, 0.01, 0.04],
                                [0.95, 0.01, 0.04],
                                [0.95, 0.01, 0.04]])
        ece, calibration_data = expected_calibration_error(y_true=y_true, y_pred_prob=y_pred_prob, n_bins=2)

        self.assertAlmostEqual(ece, 0.187, places=2)