import pandas as pd

from neat.analysis.experiment_data import ExperimentData
from neat.analysis.uncertainty.predictive_distribution import PredictionDistributionEstimator


class ExecutionsPredictionDistributions:
    def __init__(self, experiment_data: ExperimentData):
        self.experiment_data = experiment_data
        self.metrics_by_dispersion_quantile = None

    def get_metrics_by_dispersion_for_bayesian_executions(self):
        # TODO: filter those cases when there is a warning calculating F1
        data = self.experiment_data.experiment_data
        execution_ids = data.loc[data['is_bayesian'] == 1, 'execution_id'].values.tolist()
        chunks = []
        for execution_id in execution_ids:
            genome = self.experiment_data.best_genomes[execution_id]
            config = self.experiment_data.configurations[execution_id]
            predictor = PredictionDistributionEstimator(genome=genome, config=config, testing=True) \
                .estimate() \
                .enrich_with_dispersion_quantile() \
                .calculate_metrics_by_dispersion_quantile(log=False)
            metrics_by_dispersion_quantile = predictor.metrics_by_quantile
            metrics_by_dispersion_quantile['execution_id'] = execution_id
            chunks.append(metrics_by_dispersion_quantile)

        self.metrics_by_dispersion_quantile = pd.concat(chunks, sort=False, ignore_index=True)
        return self
