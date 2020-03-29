import random
import matplotlib.pyplot as plt

from config_files.configuration_utils import create_configuration
from neat.dataset.classification_mnist import MNISTDataset
from neat.dataset.classification_mnist_binary import MNISTBinaryDataset
from neat.dataset.classification_mnist_downsampled import MNISTDownsampledDataset


def main():
    dataset = MNISTDownsampledDataset(train_percentage=0.4, dataset_type='train')
    # dataset = MNISTDataset(dataset_type='test')
    # dataset = MNISTBinaryDataset(dataset_type='test')
    DATASET = 'mnist_downsampled'
    config = create_configuration(filename=f'/{DATASET}.json')
    config.n_output = 10
    dataset.generate_data()

    print(len(dataset))
    print(dataset)
    selection = random.choice(list(range(len(dataset))))
    print(selection)
    print(dataset.y)
    x, y = dataset.__getitem__(selection)
    x = x.squeeze().numpy()
    plt.imshow(x)
    plt.show()


if __name__ == '__main__':
    main()
