import random
import matplotlib.pyplot as plt

from neat.dataset.classification_mnist import MNISTDataset
from neat.dataset.classification_mnist_binary import MNISTBinaryDataset
from neat.dataset.classification_mnist_downsampled import MNISTDownsampledDataset


def main():
    dataset = MNISTDownsampledDataset(dataset_type='test')
    # dataset = MNISTDataset(dataset_type='test')
    # dataset = MNISTBinaryDataset(dataset_type='test')

    dataset.generate_data()
    print(dataset)
    selection = random.choice(list(range(len(dataset))))
    print(selection)
    x, y = dataset.__getitem__(selection)
    x = x.squeeze().numpy()
    plt.imshow(x)
    plt.show()


if __name__ == '__main__':
    main()