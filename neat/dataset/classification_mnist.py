import os

import torch
from PIL import Image
from torchvision.datasets import MNIST
from torchvision.transforms import transforms

from neat.dataset.abstract import NeatTestingDataset


class MNISTDataset(NeatTestingDataset, MNIST):

    def __init__(self, train_percentage, dataset_type='train'):
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
        NeatTestingDataset.__init__(train_percentage=train_percentage, dataset_type=dataset_type)

    def generate_data(self):
        def _data_generator(x_data: torch.Tensor):
            for i in range(len(x_data)):
                img = Image.fromarray(x_data[i].numpy(), mode='L')
                img_trans = self.transform(img)
                yield img_trans

        # self.data = self.data.float()
        self.data = torch.cat(tuple(_data_generator(x_data=self.data)), 0)
        self.targets = self.targets.long()

        self.x = self.data
        self.y = self.targets

        data_limit = self._get_data_limit()
        self.x_train = self.x[:data_limit]
        self.y_train = self.y[:data_limit]

        self.x_test = self.x[data_limit:]
        self.y_test = self.y[data_limit:]

    def __getitem__(self, item):
        return self.x[item], self.y[item]
