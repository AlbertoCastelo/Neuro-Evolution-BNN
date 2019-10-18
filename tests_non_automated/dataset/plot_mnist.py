from neat.dataset.classification_mnist import MNISTReducedDataset
import matplotlib.pyplot as plt


def main():
    dataset = MNISTReducedDataset(dataset_type='test')
    print(dataset)

    x, y = dataset.__getitem__(1)
    x = x.squeeze()
    plt.imshow(x)
    plt.show()


if __name__ == '__main__':
    main()