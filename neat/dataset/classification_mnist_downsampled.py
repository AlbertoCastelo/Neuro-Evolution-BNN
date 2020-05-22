import os

import torch
from PIL import Image
from torchvision.datasets import MNIST
from torchvision.transforms import transforms
from cv2 import resize, INTER_CUBIC, INTER_AREA

from neat.configuration import get_configuration
from neat.dataset.abstract import NeatTestingDataset
# import numpy as np
# import matplotlib.pyplot as plt


class MNISTDownsampledDataset(NeatTestingDataset, MNIST):
    DOWNSAMPLED_SIZE = (8, 8)

    def __init__(self, train_percentage, dataset_type='train', random_state=42, noise=0.0, label_noise=0.0):

        self.transform = transforms.Compose([transforms.ToTensor(),
                                             transforms.Normalize((0.1307,), (0.3081,))])
        path = ''.join([os.path.dirname(os.path.realpath(__file__)), '/data/mnist'])
        MNIST.__init__(self, root=path, train=False, download=True, transform=self.transform)
        NeatTestingDataset.__init__(self, train_percentage=train_percentage, dataset_type=dataset_type,
                                    random_state=random_state, noise=noise, label_noise=label_noise)

    def _generate_data(self):
        for output in range(get_configuration().n_output):
            mask_i = self.targets == output
            if output == 0:
                mask_or = mask_i
            else:
                mask_or += mask_i

        self.data = self.data[mask_or]
        self.targets = self.targets[mask_or]

        def _data_generator(x_data: torch.Tensor):
            for i in range(len(x_data)):
                img = x_data[i].numpy()
                img = resize(img, self.DOWNSAMPLED_SIZE, interpolation=INTER_AREA)
                img = Image.fromarray(img, mode='L')
                img_trans = self.transform(img)
                yield img_trans

        self.data = torch.cat(tuple(_data_generator(x_data=self.data)), 0)
        self.targets = self.targets.long()

        self.x = self.data
        self.y = self.targets

    def __getitem__(self, item):
        return self.x[item], self.y[item]

    def __len__(self):
        return len(self.x)
