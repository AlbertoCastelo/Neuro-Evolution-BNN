import os
from PIL import Image
from torchvision.datasets import MNIST
from torchvision.transforms import transforms

from neat.dataset.abstract import NeatTestingDataset


class MNISTDataset(NeatTestingDataset, MNIST):

    def __init__(self, dataset_type='train'):
        self.x = None
        self.y = None
        self.train = False
        if dataset_type == 'train':
            self.train = True

        # TODO: REMOVE THIS SET
        self.train = False

        self.transform = transforms.Compose([transforms.ToTensor(),
                                             transforms.Normalize((0.1307,), (0.3081,))])
        path = ''.join([os.path.dirname(os.path.realpath(__file__)), '/data/mnist'])
        MNIST.__init__(self, root=path, train=self.train, download=True, transform=self.transform)

    def generate_data(self):
        if self.train:
            self.x = self.train_data.float()
            self.y = self.train_labels.short()
        else:
            self.x = self.test_data.float()
            self.y = self.test_labels.short()
            # self.x = self.test_data
            # self.y = self.test_labels
