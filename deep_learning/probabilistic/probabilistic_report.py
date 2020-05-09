from deep_learning.probabilistic.deser import ProbabilisticFeedForwardDeser
from experiments.reporting.report import BaseReport
from experiments.reporting.report_repository import ReportRepository


class BackpropReport:
    @staticmethod
    def create(report_repository: ReportRepository, algorithm_version, dataset, correlation_id=None):
        return BackpropReport(report_repository=report_repository, algorithm_version=algorithm_version,
                              dataset=dataset, correlation_id=correlation_id)

    def __init__(self, report_repository: ReportRepository, algorithm_version, dataset, correlation_id=None):
        self.report_repository = report_repository
        self.algorithm_version = algorithm_version
        self.dataset = dataset
        self.correlation_id = correlation_id

        self.report = BaseReport(correlation_id)
        self.report.add('algorithm_version', algorithm_version)
        self.report.add('dataset', dataset)

        self.epochs = {}
        self.metrics_best = {}
        self.generic_text = None

    def set_config(self, config):
        self.report.add('configuration', config)

    def report_epoch(self, epoch, loss_train, loss_val):
        self.epochs[epoch] = {'loss_training': loss_train,
                              'loss_val': loss_val}

    def report_best_network(self, best_network, best_network_params, accuracy, f1):
        metrics_best = {'accuracy': accuracy,
                        'f1': f1}
        self.report.add('metrics', metrics_best)
        self.report.add('best_network_params', best_network_params)
        self.report.add('best_network', ProbabilisticFeedForwardDeser.to_dict(best_network))
        self.report.add('end_condition', 'normal')
        self.report.add('epochs', self.epochs)
        self.report.set_finish_time()
        return self

    def persist_report(self):
        self.report_repository.set_report(report=self.report)
        return self
