import json
from os import walk

import jsons

from experiments.file_utils import read_file
from experiments.logger import logger
from experiments.object_repository.object_repository import ObjectRepository
from experiments.object_repository.s3_utils import save_df_to_parquet_in_s3, read_df_from_parquet_in_s3
from experiments.reporting.report import BaseReport

S3_BASE_PATH_TABLES = 'tables'
S3_BASE_PATH_REPORTS = 'reports'
S3_BASE_PATH_LOGS = 'logs'


class ReportPathFactory:
    @staticmethod
    def create(algorithm_version, dataset=None, correlation_id=None, execution_id=None):
        return ReportPathFactory(algorithm_version, dataset, correlation_id, execution_id)

    def __init__(self, algorithm_version, dataset=None, correlation_id=None, execution_id=None):
        self.algorithm_version = algorithm_version
        self.execution_id = execution_id
        self.correlation_id = correlation_id
        self.dataset = dataset

    def get_algorithm_version_path(self):
        return f'version={self.algorithm_version}'

    def get_dataset_path(self):
        return f'{self.get_algorithm_version_path()}/dataset={self.dataset}'

    def get_correlation_id_path(self):
        path = f'{self.get_dataset_path()}/correlation_id={self.correlation_id}'
        return path

    def get_execution_id_path(self):
        return f'{self.get_correlation_id_path()}/execution_id={self.execution_id}'

    def get_table_path(self, table):
        return f'{S3_BASE_PATH_TABLES}/{self.get_execution_id_path()}/table={table}'

    def get_reports_path(self):
        if self.algorithm_version and self.dataset and self.correlation_id:
            return f'{S3_BASE_PATH_REPORTS}/{self.get_correlation_id_path()}'
        elif self.algorithm_version and self.dataset:
            return f'{S3_BASE_PATH_REPORTS}/{self.get_dataset_path()}'
        elif self.algorithm_version:
            return f'{S3_BASE_PATH_REPORTS}/{self.get_algorithm_version_path()}'
        raise ValueError()

    def get_report_path(self):
        data_path = f'{S3_BASE_PATH_REPORTS}/{self.get_execution_id_path()}'
        return data_path

    def get_logs_path(self):
        data_path = f'{S3_BASE_PATH_LOGS}/version={self.algorithm_version}/dataset={self.dataset}/' \
                    f'correlation_id={self.correlation_id}/' \
                    f'execution_id={self.execution_id}'
        return data_path


class ReportRepository:
    @staticmethod
    def create(project: str, logs_path: str, object_repository: ObjectRepository = None):
        if object_repository is None:
            bucket = ReportRepository.build_bucket_name(project)
            object_repository = ObjectRepository(bucket=bucket, logger=logger)
            if not object_repository.bucket_exists():
                object_repository.create_bucket()

        return ReportRepository(project=project, logs_path=logs_path, object_repository=object_repository)

    def __init__(self, project: str, logs_path: str, object_repository: ObjectRepository):
        self.project = project
        self.bucket = None
        self.object_repository = object_repository

    def get_executions(self, algorithm_version, dataset, correlation_id):
        path = ReportPathFactory.create(algorithm_version=algorithm_version,
                                        dataset=dataset,
                                        correlation_id=correlation_id)\
            .get_reports_path()
        for execution in self.object_repository.tree(path):
            yield execution.split('/')[0].split('=')[1]

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

    def persist_logs(self, algorithm_version, dataset, correlation_id, execution_id):
        object_store_path = ReportPathFactory.create(algorithm_version=algorithm_version,
                                                     dataset=dataset,
                                                     correlation_id=correlation_id,
                                                     execution_id=execution_id) \
            .get_logs_path()
        # get all files in path
        log_path = logger.log_base_path
        for (dirpaths, dirnames, filenames) in walk(log_path):
            for filename in filenames:
                key = object_store_path + '/' + filename
                filename_complete = log_path + '/' + filename

                data = read_file(filename=filename_complete)
                self.object_repository.set(key=key, content=data)
                # self.object_repository.set_from_file(key=,
                #                                      data_path=)

    def download_logs(self, algorithm_version, dataset, correlation_id, execution_id, local_log_path='./'):
        object_store_path = ReportPathFactory.create(algorithm_version=algorithm_version,
                                                     dataset=dataset,
                                                     correlation_id=correlation_id,
                                                     execution_id=execution_id) \
            .get_logs_path()
        remote_files = self.object_repository.tree(key=object_store_path)
        # get all files in path
        for remote_file in remote_files:
            self.object_repository.get_to_file(key=object_store_path + '/' + remote_file,
                                               data_path=local_log_path + '/' + remote_file)

    @staticmethod
    def build_bucket_name(project):
        return f'{project}'
