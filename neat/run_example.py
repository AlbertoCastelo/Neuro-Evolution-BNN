import sys
import random

sys.path.append('./')

from config_files.configuration_utils import create_configuration
from experiments.reporting.report_repository import ReportRepository
from experiments.slack_client import SlackNotifier
from neat.neat_logger import get_neat_logger
import os

from neat.population_engine import EvolutionEngine
from neat.reporting.reports_pyneat import EvolutionReport
import fire

node_add_probs = [0.3, 0.5, 0.7]
connection_add_probs = [0.3, 0.5, 0.7]


class ExecutionRunner:
    def run(self, dataset_name, algorithm_version, correlation_id, config_parameters):
        print(config_parameters)
        config = create_configuration(filename=f'/{dataset_name}.json')
        config = self._modify_config(config, config_parameters)
        config.dataset_random_state = random.sample(list(range(100)), k=1)[0]
        LOGS_PATH = f'{os.getcwd()}/'
        logger = get_neat_logger(path=LOGS_PATH)

        report_repository = ReportRepository.create(project='neuro-evolution', logs_path=LOGS_PATH)
        notifier = SlackNotifier.create(channel='batch-jobs')

        try:
            report = EvolutionReport(report_repository=report_repository,
                                     algorithm_version=algorithm_version,
                                     dataset=dataset_name,
                                     correlation_id=correlation_id)
            notifier.send(f'New job using: node_add_prob={config_parameters}')
            print(report.report.execution_id)
            evolution_engine = EvolutionEngine(report=report, notifier=notifier)
            evolution_engine.run()
        except Exception as e:
            print(e)
            notifier.send(str(e))
            logger.error(str(e))

    def _modify_config(self, config, config_parameters):
        for name, value in config_parameters.items():
            if hasattr(config, name):
                setattr(config, name, value)
            else:
                raise ValueError(f'Attribute: {name} does not exists')
        return config


if __name__ == '__main__':
    fire.Fire(ExecutionRunner)
