from unittest import TestCase
from unittest.mock import Mock
import os

from pandas import DataFrame

from config_files.configuration_utils import create_configuration
from deep_learning.nas import neural_architecture_search, ALGORITHM_VERSION
from deep_learning.probabilistic.evaluate_probabilistic_dl import EvaluateProbabilisticDL
from deep_learning.standard.evaluate_standard_dl import EvaluateStandardDL
from experiments.reporting.report_repository import ReportRepository
from neat.analysis.experiment_data import ExperimentDataNAS
from neat.neat_logger import get_neat_logger

LOGS_PATH = f'{os.getcwd()}/'
logger = get_neat_logger(path=LOGS_PATH)


class TestNASIntegration(TestCase):
    def setUp(self) -> None:
        self.config = create_configuration(filename='/classification-miso.json')

    def test_nas_standard_integrate_with_report(self):
        correlation_id_standard = 'test_standard'
        correlation_id_probabilistic = 'test_probabilistic'
        project = 'test'
        notifier = Mock()

        report_repository = ReportRepository.create(project='test', logs_path=LOGS_PATH)

        backprop_report_standard = neural_architecture_search(EvaluateDL=EvaluateStandardDL,
                                                              n_hidden_layers_values=[1],
                                                              n_neurons_per_layer_values=[1],
                                                              correlation_id=correlation_id_standard,
                                                              config=self.config,
                                                              batch_size=1000000,
                                                              lr=0.01,
                                                              weight_decay=0.0005,
                                                              n_epochs=1,
                                                              notifier=notifier,
                                                              report_repository=report_repository,
                                                              is_cuda=False,
                                                              n_repetitions=1)

        backprop_report_probabilistic = neural_architecture_search(EvaluateDL=EvaluateProbabilisticDL,
                                                                   n_hidden_layers_values=[1],
                                                                   n_neurons_per_layer_values=[1],
                                                                   correlation_id=correlation_id_probabilistic,
                                                                   config=self.config,
                                                                   batch_size=1000000,
                                                                   lr=0.01,
                                                                   weight_decay=0.0005,
                                                                   n_epochs=1,
                                                                   notifier=notifier,
                                                                   report_repository=report_repository,
                                                                   is_cuda=False,
                                                                   n_repetitions=1)

        experiment_data_nas = ExperimentDataNAS(correlation_ids=[correlation_id_standard,correlation_id_probabilistic],
                                                dataset_name=self.config.dataset,
                                                n_samples=10,
                                                project=project,
                                                algorithm_version=ALGORITHM_VERSION,
                                                keep_top=1.0,
                                                filter_checkpoint_finish=False)\
                # .process_data()
        reports_all = experiment_data_nas._get_reports()
        reports = {}
        for execution_id, report in reports_all.items():
            if execution_id in [backprop_report_standard.report.execution_id,
                                backprop_report_probabilistic.report.execution_id]:

                reports[execution_id] = report

        data_df = experiment_data_nas._process_reports(reports)

        self.assertEqual(type(data_df), DataFrame)

        expected_accuracy = data_df.loc[(data_df['correlation_id'] == correlation_id_standard) &
                                        (data_df['execution_id'] == backprop_report_standard.report.execution_id),
                                         'accuracy'].values[0]
        # self.assertEqual(backprop_report_standard.report.metrics['accuracy'], expected_accuracy)
