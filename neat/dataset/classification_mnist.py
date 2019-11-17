import os
from PIL import Image
from torchvision.datasets import MNIST
from torchvision.transforms import transforms

from neat.dataset.abstract import NeatTestingDataset


class MNISTReducedDataset(NeatTestingDataset, MNIST):

    def __init__(self, dataset_type='train'):
        self.train = False
        if dataset_type == 'train':
            self.train = True

        self.transform = transforms.Compose([transforms.ToTensor(),
                                             transforms.Normalize((0.1307,), (0.3081,))])
        path = os.path.realpath(__file__)
        MNIST.__init__(self, root='/', train=self.train, download=True, transform=self.transform)

    def generate_data(self):
        pass

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        if self.train:
            img, target = self.train_data[index], self.train_labels[index]
        else:
            img, target = self.test_data[index], self.test_labels[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img.numpy(), mode='L')

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target
