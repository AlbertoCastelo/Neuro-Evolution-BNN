import torch
from torch.utils.data import Dataset
import numpy as np


class RegressionExample1Dataset(Dataset):
    '''
    Dataset with 1 input variables and 1 output
    '''
    def __init__(self, dataset_type='train', is_debug=False):
        if dataset_type not in ['train', 'validation', 'test']:
            raise ValueError(f'Dataset Type {dataset_type} is not valid')

        self.dataset_type = dataset_type
        if dataset_type == 'train':
            is_training = True
        self.is_debug = is_debug

        if is_training:
            range_ = [0.0, 0.7]
            noise = [0.0, 0.02]

        self.dataset_size = 500
        x = np.random.uniform(range_[0], range_[1], self.dataset_size)
        y = x + 0.3*np.sin(2*np.pi*(x+np.random.normal(noise[0], noise[1], self.dataset_size))) + \
            0.3 * np.sin(4 * np.pi * (x + np.random.normal(noise[0], noise[1], self.dataset_size)))+ \
            np.random.normal(noise[0], noise[1], self.dataset_size)

        self.x = x
        self.y = y
        if is_debug:
            self.x = x[:512]
            self.y = y[:512]

        # create training/validation set
        # train, validation_and_test = train_test_split(self.df_data, random_state=42, shuffle=True, test_size=0.4)
        # validation, test = train_test_split(validation_and_test, random_state=42, shuffle=True, test_size=0.5)
        #
        # self.df_data
        # if self.dataset_type == 'train':
        #     self.df_data = train
        # elif self.dataset_type == 'validation':
        #     self.df_data = validation
        # elif self.dataset_type == 'test':
        #     self.df_data = test
        # else:
        #     raise ValueError

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        x = torch.tensor(self.x[idx])
        y = torch.tensor(self.y[idx])
        return x, y


class RegressionExample2Dataset(Dataset):
    '''
    Dataset with 2 input variables and 1 output
    '''
    def __init__(self, dataset_type='train', is_debug=False):
        if dataset_type not in ['train', 'validation', 'test']:
            raise ValueError(f'Dataset Type {dataset_type} is not valid')

        self.dataset_type = dataset_type
        if dataset_type == 'train':
            is_training = True
        self.is_debug = is_debug

        if is_training:
            range_ = [0.0, 0.7]
            noise = [0.0, 0.02]

        self.dataset_size = 500
        x_1 = np.random.uniform(range_[0], range_[1], self.dataset_size)
        x_2 = np.random.uniform(range_[0], range_[1], self.dataset_size)
        y = x_1 + 0.3*np.sin(2*np.pi*(x_1+np.random.normal(noise[0], noise[1], self.dataset_size))) + \
            0.3 * np.sin(4 * np.pi * (x_2 + np.random.normal(noise[0], noise[1], self.dataset_size))) + \
            np.random.normal(noise[0], noise[1], self.dataset_size)

        self.x = np.array(list(zip(x_1,x_2)))
        self.y = y
        if is_debug:
            self.x = self.x[:512]
            self.y = self.y[:512]

        # create training/validation set
        # train, validation_and_test = train_test_split(self.df_data, random_state=42, shuffle=True, test_size=0.4)
        # validation, test = train_test_split(validation_and_test, random_state=42, shuffle=True, test_size=0.5)
        #
        # self.df_data
        # if self.dataset_type == 'train':
        #     self.df_data = train
        # elif self.dataset_type == 'validation':
        #     self.df_data = validation
        # elif self.dataset_type == 'test':
        #     self.df_data = test
        # else:
        #     raise ValueError

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        x = torch.tensor(self.x[idx])
        y = torch.tensor(self.y[idx])
        return x, y
