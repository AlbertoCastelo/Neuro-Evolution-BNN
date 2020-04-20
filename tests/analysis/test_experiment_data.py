from unittest import TestCase
import pandas as pd
from pandas.util.testing import assert_frame_equal

from neat.analysis.experiment_data import ExperimentData


class TestExperimentData(TestCase):

    def test_drop_worse_executions_per_correlation(self):
        experiment_data = pd.DataFrame({'correlation_id': [1, 1, 1, 2, 2],
                                        'execution_id': list(range(5)),
                                        'loss_training': [0.9, 0.8, 0.1, 0.7, 0.5]})

        expected_experiment_data = pd.DataFrame({'correlation_id': [1, 1, 2],
                                                 'execution_id': [2, 1, 4],
                                                 'loss_training': [0.1, 0.8, 0.5]})
        drop_worst = 0.5
        experiment_data_filtered = ExperimentData._drop_worse_executions_per_correlation(experiment_data, drop_worst)

        assert_frame_equal(expected_experiment_data, experiment_data_filtered)
