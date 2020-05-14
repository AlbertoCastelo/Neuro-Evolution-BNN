from neat.configuration import ConfigError
from neat.dataset.classification_cancer import HistoPathologicCancer
from neat.dataset.classification_example import ClassificationExample1Dataset
from neat.dataset.classification_example_3 import ClassificationExample2Dataset
from neat.dataset.classification_mnist import MNISTDataset
from neat.dataset.classification_mnist_binary import MNISTBinaryDataset
from neat.dataset.classification_mnist_downsampled import MNISTDownsampledDataset
from neat.dataset.classification_titanic import TitanicDataset
from neat.dataset.classification_xray_binary import XRayBinary
from neat.dataset.iris import IrisDataset
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


def get_dataset(dataset, train_percentage=0.4, testing=False, random_state=42, noise=0.0, label_noise=0.0):
    if testing:
        dataset_type = 'test'
    else:
        dataset_type = 'train'

    if dataset == 'regression-siso':
        dataset = RegressionExample1Dataset(train_percentage=train_percentage, dataset_type=dataset_type,
                                            random_state=random_state, noise=noise)
    elif dataset == 'regression-miso':
        dataset = RegressionExample2Dataset(train_percentage=train_percentage, dataset_type=dataset_type,
                                            random_state=random_state, noise=noise)
    elif dataset == 'classification-miso':
        dataset = ClassificationExample1Dataset(train_percentage=train_percentage, dataset_type=dataset_type,
                                                random_state=random_state, noise=noise)
    elif dataset == 'classification-miso-3':
        dataset = ClassificationExample2Dataset(train_percentage=train_percentage, dataset_type=dataset_type,
                                                random_state=random_state, noise=noise)
    elif dataset == 'titanic':
        dataset = TitanicDataset(train_percentage=train_percentage, dataset_type=dataset_type,
                                 random_state=random_state, noise=noise, label_noise=label_noise)
    elif dataset == 'mnist':
        dataset = MNISTDataset(train_percentage=train_percentage, dataset_type=dataset_type,
                               random_state=random_state, noise=noise)
    elif dataset == 'mnist_downsampled':
        dataset = MNISTDownsampledDataset(train_percentage=train_percentage, dataset_type=dataset_type,
                                          random_state=random_state, noise=noise)
    elif dataset == 'mnist_binary':
        dataset = MNISTBinaryDataset(train_percentage=train_percentage, dataset_type=dataset_type,
                                     random_state=random_state, noise=noise)
    elif dataset == 'xray_binary':
        dataset = XRayBinary(train_percentage=train_percentage, dataset_type=dataset_type,
                             random_state=random_state, noise=noise)
    elif dataset == 'cancer':
        dataset = HistoPathologicCancer(train_percentage=train_percentage, dataset_type=dataset_type,
                                        random_state=random_state, noise=noise)
    elif dataset == 'iris':
        dataset = IrisDataset(train_percentage=train_percentage, dataset_type=dataset_type,
                              random_state=random_state, noise=noise, label_noise=label_noise)
    else:
        raise ConfigError(f'Dataset Name is incorrect: {dataset}')
    dataset.generate_data()
    print(f'Training: {len(dataset.x_train)}. Testing: {len(dataset.x_test)}')
    return dataset
