from experiments.reporting.report_repository import ReportRepository

ALGORITHM_VERSION = 'bayes-neat'
DATASET = 'toy-classification'
CORRELATION_ID = 'parameters_grid'


def main():
    execution_id = '4cfdcb16-bd67-485b-ac48-107199211b46'
    report_repository = ReportRepository.create(project='neuro-evolution')
    report = report_repository.get_report(algorithm_version=ALGORITHM_VERSION, dataset=DATASET,
                                          correlation_id=CORRELATION_ID, execution_id=execution_id)
    print(report.__dict__)


if __name__ == '__main__':
    main()
