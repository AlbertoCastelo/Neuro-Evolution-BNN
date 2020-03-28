from neat.configuration import ConfigError
from neat.dataset.classification_example import ClassificationExample1Dataset
from neat.dataset.classification_example_3 import ClassificationExample2Dataset
from neat.dataset.classification_mnist import MNISTDataset
from neat.dataset.classification_mnist_binary import MNISTBinaryDataset
from neat.dataset.classification_mnist_downsampled import MNISTDownsampledDataset
from neat.dataset.classification_titanic import TitanicDataset
from neat.dataset.regression_example import RegressionExample1Dataset, RegressionExample2Dataset


def _prepare_batch_data(x_batch, y_batch, is_gpu, n_input, n_output, problem_type, n_samples):
    x_batch = x_batch.view(-1, n_input).repeat(n_samples, 1)

    if problem_type == 'classification':
        y_batch = y_batch.view(-1, 1).repeat(n_samples, 1).squeeze()
    elif problem_type == 'regression':
        y_batch = y_batch.view(-1, n_output).repeat(n_samples, 1)
    else:
        raise ValueError(f'Problem Type is not correct: {problem_type}')

    if is_gpu:
        x_batch, y_batch = x_batch.cuda(), y_batch.cuda()

    return x_batch, y_batch


def get_dataset(dataset, train_percentage=0.4, testing=False):
    if testing:
        dataset_type = 'test'
    else:
        dataset_type = 'train'

    if dataset == 'regression_example_1':
        dataset = RegressionExample1Dataset(dataset_type=dataset_type)
    elif dataset == 'regression_example_2':
        dataset = RegressionExample2Dataset(dataset_type=dataset_type)
    elif dataset == 'classification-miso':
        dataset = ClassificationExample1Dataset(dataset_type=dataset_type)
    elif dataset == 'classification-miso-3':
        dataset = ClassificationExample2Dataset(dataset_type=dataset_type)
    elif dataset == 'titanic':
        dataset = TitanicDataset(train_percentage=train_percentage, dataset_type=dataset_type)
    elif dataset == 'mnist':
        dataset = MNISTDataset(dataset_type=dataset_type)
    elif dataset == 'mnist_downsampled':
        dataset = MNISTDownsampledDataset(dataset_type=dataset_type)
    elif dataset == 'mnist_binary':
        dataset = MNISTBinaryDataset(dataset_type=dataset_type)
    else:
        raise ConfigError(f'Dataset Name is incorrect: {dataset}')
    dataset.generate_data()
    return dataset
