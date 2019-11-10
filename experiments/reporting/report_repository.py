import json

import jsons

from experiments.logger import logger
from experiments.object_repository.object_repository import ObjectRepository
from experiments.object_repository.s3_utils import save_df_to_parquet_in_s3, read_df_from_parquet_in_s3
from experiments.reporting.report import BaseReport

S3_BASE_PATH_TABLES = 'tables'
S3_BASE_PATH_REPORTS = 'reports'


class ReportPathFactory:
    @staticmethod
    def create(algorithm_version, dataset, execution_id, correlation_id):
        return ReportPathFactory(algorithm_version, dataset, execution_id, correlation_id)

    def __init__(self, algorithm_version, dataset, execution_id, correlation_id):
        self.algorithm_version = algorithm_version
        self.execution_id = execution_id
        self.correlation_id = correlation_id
        self.dataset = dataset

    def get_table_path(self, table):
        data_path = f'{S3_BASE_PATH_TABLES}/version={self.algorithm_version}/dataset={self.dataset}/' \
                    f'correlation_id={self.correlation_id}/' \
                    f'execution_id={self.execution_id}/table={table}'
        return data_path

    def get_report_path(self):
        data_path = f'{S3_BASE_PATH_REPORTS}/version={self.algorithm_version}/dataset={self.dataset}/' \
                    f'correlation_id={self.correlation_id}/' \
                    f'execution_id={self.execution_id}'
        return data_path


class ReportRepository:
    @staticmethod
    def create(project: str, object_repository: ObjectRepository = None):
        if object_repository is None:
            bucket = ReportRepository.build_bucket_name(project)
            object_repository = ObjectRepository(bucket=bucket, logger=logger)
            if not object_repository.bucket_exists():
                object_repository.create_bucket()

        return ReportRepository(project=project, object_repository=object_repository)

    def __init__(self, project: str, object_repository: ObjectRepository):
        self.project = project
        self.bucket = None
        self.object_repository = object_repository

    def set_table(self, algorithm_version, dataset, execution_id, correlation_id, table_name, table_value):
        path = ReportPathFactory.create(algorithm_version=algorithm_version,
                                        dataset=dataset,
                                        correlation_id=correlation_id,
                                        execution_id=execution_id) \
            .get_table_path(table_name)
        save_df_to_parquet_in_s3(df=table_value, bucket=self.bucket, data_path=path, name=table_name,
                                 chunks_size=1000000000)

    def get_table(self, algorithm_version, dataset, execution_id, correlation_id, table_name, table_value):
        path = ReportPathFactory.create(algorithm_version=algorithm_version,
                                        dataset=dataset,
                                        correlation_id=correlation_id,
                                        execution_id=execution_id) \
            .get_table_path(table_name)
        read_df_from_parquet_in_s3(bucket=self.bucket, data_path=path)

    def set_report(self, report: BaseReport):
        algorithm_version = report.get('algorithm_version')
        dataset = report.get('dataset')
        execution_id = report.get('execution_id')
        correlation_id = report.get('correlation_id')
        key = self._get_report_key(algorithm_version, correlation_id, dataset, execution_id)
        data = json.dumps(report.to_dict())

        self.object_repository.set(key=key, content=str(data))

    def _get_report_key(self, algorithm_version, correlation_id, dataset, execution_id):
        path = ReportPathFactory.create(algorithm_version=algorithm_version,
                                        dataset=dataset,
                                        correlation_id=correlation_id,
                                        execution_id=execution_id) \
            .get_report_path()
        key = f'{path}/report.json'
        return key

    def get_report(self, algorithm_version, dataset, correlation_id, execution_id) -> BaseReport:
        key = self._get_report_key(algorithm_version, correlation_id, dataset, execution_id)
        report_dict = json.loads(self.object_repository.get(key))
        return BaseReport.from_dict(report_dict)

    @staticmethod
    def build_bucket_name(project):
        return f'{project}'
