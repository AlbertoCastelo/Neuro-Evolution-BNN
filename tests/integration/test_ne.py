from unittest import TestCase
from unittest.mock import Mock
import os

from pandas import DataFrame

from config_files.configuration_utils import create_configuration
from deep_learning.nas import neural_architecture_search, ALGORITHM_VERSION
from deep_learning.probabilistic.evaluate_probabilistic_dl import EvaluateProbabilisticDL
from deep_learning.standard.evaluate_standard_dl import EvaluateStandardDL
from experiments.reporting.report_repository import ReportRepository
from neat.analysis.experiment_data import ExperimentDataNAS, ExperimentDataNE
from neat.neat_logger import get_neat_logger
from neat.population_engine import EvolutionEngine
from neat.reporting.reports_pyneat import EvolutionReport

LOGS_PATH = f'{os.getcwd()}/'
logger = get_neat_logger(path=LOGS_PATH)


class TestNEIntegration(TestCase):
    def setUp(self) -> None:
        self.config = create_configuration(filename='/classification-miso.json')
        self.config.pop_size = 20
        self.config.is_fine_tuning = True
        self.config.epochs_fine_tuning = 1
        self.config.n_generations = 1

    def test_nas_standard_integrate_with_report(self):
        correlation_id = 'test_probabilistic'
        algorithm_version = 'test_neat'
        project = 'test-ne'
        notifier = Mock()

        report_repository = ReportRepository.create(project=project, logs_path=LOGS_PATH)

        report_ne = EvolutionReport(report_repository=report_repository,
                                    algorithm_version=algorithm_version,
                                    dataset=self.config.dataset,
                                    correlation_id=correlation_id)
        evolution_engine = EvolutionEngine(report=report_ne, notifier=notifier, is_cuda=False)
        evolution_engine.run()

        experiment_data_ne = ExperimentDataNE(correlation_ids=[correlation_id],
                                              dataset_name=self.config.dataset,
                                              n_samples=10,
                                              project=project,
                                              algorithm_version=algorithm_version,
                                              keep_top=1.0,
                                              filter_checkpoint_finish=False)

        reports_all = experiment_data_ne._get_reports()
        reports = {}
        for execution_id, report in reports_all.items():
            if execution_id == report_ne.report.execution_id:
                reports[execution_id] = report

        data_df = experiment_data_ne._process_reports(reports)

        self.assertEqual(type(data_df), DataFrame)

        # expected_accuracy = data_df.loc[(data_df['correlation_id'] == correlation_id_standard) &
        #                                 (data_df['execution_id'] == backprop_report_standard.report.execution_id),
        #                                  'accuracy'].values[0]
        # self.assertEqual(backprop_report_standard.report.metrics['accuracy'], expected_accuracy)
