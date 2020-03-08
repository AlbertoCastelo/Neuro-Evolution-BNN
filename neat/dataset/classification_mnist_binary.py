import os

import torch
from PIL import Image
from torchvision.datasets import MNIST
from torchvision.transforms import transforms

from neat.dataset.abstract import NeatTestingDataset


class MNISTBinaryDataset(NeatTestingDataset, MNIST):
    '''
    MNIST dataset considering only 2 classes: 1 and 2 digits.
    '''
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
        mask_1 = self.targets == 1
        mask_0 = self.targets == 0
        mask_or = mask_1 + mask_0
        self.data = self.data[mask_or]
        self.targets = self.targets[mask_or]

        def _data_generator(x_data: torch.Tensor):
            for i in range(len(x_data)):
                img = Image.fromarray(x_data[i].numpy(), mode='L')
                img_trans = self.transform(img)
                yield img_trans

        # self.data = self.data.float()
        self.data = torch.cat(tuple(_data_generator(x_data=self.data)), 0)
        # self.data = self.transform(self.data.numpy())
        self.targets = self.targets.long()

        # print(f'data: {self.data.shape}')

        self.x = self.data
        self.y = self.targets

    def __getitem__(self, item):
        return self.x[item], self.y[item]
