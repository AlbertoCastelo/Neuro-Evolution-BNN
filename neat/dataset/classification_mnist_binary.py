import os

import torch
from PIL import Image
from torchvision.datasets import MNIST
from torchvision.transforms import transforms

from neat.configuration import get_configuration
from neat.dataset.abstract import NeatTestingDataset

n_output = get_configuration().n_output


class MNISTBinaryDataset(NeatTestingDataset, MNIST):
    '''
    MNIST dataset considering only 2 classes: 1 and 2 digits.
    '''
    def __init__(self, train_percentage, dataset_type='train', random_state=42):
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
        MNIST.__init__(self, root=path, train=False, download=True, transform=self.transform)
        NeatTestingDataset.__init__(self, train_percentage=train_percentage, dataset_type=dataset_type,
                                    random_state=random_state)

    def generate_data(self):
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
                img = Image.fromarray(x_data[i].numpy(), mode='L')
                img_trans = self.transform(img)
                yield img_trans

        self.data = torch.cat(tuple(_data_generator(x_data=self.data)), 0)
        self.targets = self.targets.long()

        self.x = self.data
        self.y = self.targets

        self._generate_train_test_sets()
        # data_limit = self._get_data_limit()
        # self.x_train = self.x[:data_limit]
        # self.y_train = self.y[:data_limit]
        #
        # self.x_test = self.x[data_limit:]
        # self.y_test = self.y[data_limit:]

    def __getitem__(self, item):
        return self.x[item], self.y[item]

    def __len__(self):
        return len(self.x)
