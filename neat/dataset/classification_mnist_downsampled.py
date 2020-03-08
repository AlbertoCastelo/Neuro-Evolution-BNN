import os

import torch
from PIL import Image
from torchvision.datasets import MNIST
from torchvision.transforms import transforms
from cv2 import resize, INTER_CUBIC, INTER_AREA
from neat.dataset.abstract import NeatTestingDataset
import numpy as np
import matplotlib.pyplot as plt

class MNISTDownsampledDataset(NeatTestingDataset, MNIST):
    DOWNSAMPLED_SIZE = (16, 16)

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
        def _data_generator(x_data: torch.Tensor):
            for i in range(len(x_data)):
                img = x_data[i].numpy()
                img = resize(img, self.DOWNSAMPLED_SIZE, interpolation=INTER_AREA)
                img = Image.fromarray(img, mode='L')
                img_trans = self.transform(img)
                yield img_trans

        self.data = torch.cat(tuple(_data_generator(x_data=self.data)), 0)
        # self.data = self.data.float()
        self.targets = self.targets.long()

        self.x = self.data
        self.y = self.targets

    def __getitem__(self, item):
        return self.x[item], self.y[item]
