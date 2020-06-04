import pandas as pd

from neat.analysis.experiment_data import ExperimentData
from neat.analysis.uncertainty.predictive_distribution import PredictionDistributionEstimator, \
    PredictionDistributionEstimatorGenome


class ExecutionsPredictionDistributions:
    def __init__(self, experiment_data: ExperimentData):
        self.experiment_data = experiment_data
        self.metrics_by_dispersion_quantile = None
        self.prediction_distribution_estimators = {}

    def run(self, testing=True, filter_no_bayesian=True):
        # TODO: filter those cases when there is a warning calculating F1
        data = self.experiment_data.experiment_data
        if filter_no_bayesian:
            execution_ids = data.loc[data['is_bayesian'] == 1, 'execution_id'].unique().tolist()
        else:
            execution_ids = data['execution_id'].unique().tolist()
        chunks = []
        for execution_id in execution_ids:
            genome = self.experiment_data.best_genomes[execution_id]
            config = self.experiment_data.configurations[execution_id]
            predictor = PredictionDistributionEstimatorGenome(genome=genome, config=config, testing=testing) \
                .estimate() \
                .enrich_with_dispersion_quantile() \
                .calculate_metrics_by_dispersion_quantile(log=False)
            metrics_by_dispersion_quantile = predictor.metrics_by_quantile
            metrics_by_dispersion_quantile['execution_id'] = execution_id
            chunks.append(metrics_by_dispersion_quantile)
            self.prediction_distribution_estimators[execution_id] = predictor

        self.metrics_by_dispersion_quantile = pd.concat(chunks, sort=False, ignore_index=True)
        return self

    def get_metrics_by_dispersion_quantile(self) -> pd.DataFrame:
        return self.metrics_by_dispersion_quantile

    def get_prediction_distribution_estimators(self) -> dict:
        return self.prediction_distribution_estimators
