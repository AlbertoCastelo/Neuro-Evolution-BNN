from datetime import datetime, timezone


class BaseReport:
    def __init__(self, experiment_name: str):
        self.experiment_name = experiment_name
        experiment_datetime = datetime.now(timezone.utc).strftime("%m-%d-%Y, %H:%M:%S")
        self.execution_id = f'{experiment_datetime}__{experiment_name}'
