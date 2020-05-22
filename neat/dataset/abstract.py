import random

from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
import torch
import numpy as np


class NeatTestingDataset(Dataset):
    def __init__(self, train_percentage, dataset_type, random_state, noise, label_noise):
        self.train_percentage = train_percentage
        self.dataset_type = dataset_type
        self.random_state = random_state
        self.noise = noise
        self.label_noise = label_noise

        self.train = False
        self.x = None
        self.y = None
        self.x_train = None
        self.y_train = None
        self.x_test = None
        self.y_test = None
        if dataset_type == 'train':
            self.train = True

    def _generate_train_test_sets(self):
        x_train, x_test, y_train, y_test = train_test_split(self.x.numpy(), self.y.numpy(),
                                                            test_size=1 - self.train_percentage,
                                                            random_state=self.random_state)

        self.x_train = torch.tensor(x_train)
        self.y_train = torch.tensor(y_train)
        self.x_test = torch.tensor(x_test)
        self.y_test = torch.tensor(y_test)

        print(f'Sum Train: {self.x_train.sum()}')

    def generate_data(self):
        self._generate_data()

        self._generate_train_test_sets()

        self._add_noise_to_train_labels(self.label_noise)

    def _get_data_limit(self):
        return int(round(len(self.x) * self.train_percentage))

    def _generate_data(self):
        raise NotImplementedError

    def _add_noise(self, x):
        if self.noise > 0:
            means = np.mean(x, 0)
            stds = np.std(x, 0)
            x_noisy = ((x - means)/stds +
                       np.random.normal(0, self.noise, size=x.shape)
                       + means) * stds
            return x_noisy
        return x

    def _add_noise_to_train_labels(self, label_noise):
        '''
        :param label_noise: percentage of labels flipped
        '''

        y_train = self.y_train.numpy()
        n_classes = len(set(y_train))
        n_examples = len(y_train)
        class_choices = set(range(n_classes))

        print(f'Label Noise: {label_noise}')
        random.seed(self.random_state)
        for i in range(n_examples):
            r = random.random()
            if r < label_noise:
                y_train[i] = random.choice(list(class_choices - {y_train[i]}))
        print(f'Sum of 10 first labels: {np.sum(y_train[:10])}')
        print(y_train[:10])
        self.y_train = torch.tensor(y_train)
