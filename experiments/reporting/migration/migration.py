import sys
sys.path.append('./')

import fire
from experiments.reporting.report_repository import ReportRepository


class Migration:
    def download_reports(self, project, algorithm_version, dataset, correlation_id, base_dir='./'):
        report_repository = ReportRepository.create(project=project, logs_path=None)
        report_repository.download_reports_to_dir(algorithm_version, dataset, correlation_id, base_dir=base_dir)

    def upload_reports(self, project, algorithm_version, dataset, correlation_id, new_correlation_id, base_dir='./'):
        report_repository = ReportRepository.create(project=project, logs_path=None)
        report_repository.upload_reports_from_dir(algorithm_version, dataset, correlation_id, new_correlation_id,
                                                  base_dir=base_dir)



if __name__ == '__main__':
    fire.Fire(Migration)
