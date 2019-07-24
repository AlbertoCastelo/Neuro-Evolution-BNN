import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset
import numpy as np


class RegressionExample1Dataset(Dataset):
    '''
    Dataset with 1 input variables and 1 output
    '''
    TRAIN_SIZE = 500
    TEST_SIZE = 500

    def __init__(self, dataset_type='train', is_debug=False):
        self.is_debug = is_debug

        self.input_scaler = StandardScaler()
        self.output_scaler = StandardScaler()

        if dataset_type not in ['train', 'validation', 'test']:
            raise ValueError(f'Dataset Type {dataset_type} is not valid')

        range_ = [0.0, 0.7]
        noise = [0.0, 0.02]
        x_train = np.random.uniform(range_[0], range_[1], self.TRAIN_SIZE)
        x_train, y_train = self._get_x_y(x=x_train, noise=noise)

        self.input_scaler.fit(x_train)
        self.output_scaler.fit(y_train)

        if dataset_type == 'train':
            x = x_train
            y = y_train
        if dataset_type == 'test':
            x_test = np.linspace(-0.5, 1.5, self.TEST_SIZE)
            noise = [0.00, 0.00]
            x, y = self._get_x_y(x=x_test, noise=noise)

        self.x_original = x
        self.y_original = y
        self.x = self.input_scaler.transform(x)
        self.y = self.output_scaler.transform(y).reshape(-1)

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

    def _get_x_y(self, x, noise):
        dataset_size = x.shape[0]

        y = x + 0.3 * np.sin(2 * np.pi * (x + np.random.normal(noise[0], noise[1], dataset_size))) + \
            0.3 * np.sin(4 * np.pi * (x + np.random.normal(noise[0], noise[1], dataset_size))) + \
            np.random.normal(noise[0], noise[1], )
        return x.reshape((-1, 1)), y.reshape((-1, 1))

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        x = torch.tensor(self.x[idx])
        y = torch.tensor(self.y[idx])
        return x, y


class RegressionExample2Dataset(Dataset):
    TRAIN_SIZE = 500
    TEST_SIZE = 500
    '''
    Dataset with 2 input variables and 1 output
    '''
    def __init__(self, dataset_type='train', is_debug=False):
        self.input_scaler = StandardScaler()
        self.output_scaler = StandardScaler()
        if dataset_type not in ['train', 'validation', 'test']:
            raise ValueError(f'Dataset Type {dataset_type} is not valid')

        range_ = [0.0, 0.7]
        noise = [0.0, 0.02]
        x_1_train = np.random.uniform(range_[0], range_[1], self.TRAIN_SIZE)
        x_2_train = np.random.uniform(range_[0], range_[1], self.TRAIN_SIZE)
        x_train, y_train = self._get_x_y(x_1=x_1_train, x_2=x_2_train, noise=noise)

        self.input_scaler.fit(x_train)
        self.output_scaler.fit(y_train)

        if dataset_type == 'train':
            x = x_train
            y = y_train
        if dataset_type == 'test':
            x_1_test = np.linspace(-0.5, 1.5, self.TEST_SIZE)
            x_2_test = np.linspace(-0.5, 1.5, self.TEST_SIZE)
            noise = [0.00, 0.00]
            x, y = self._get_x_y(x_1=x_1_test, x_2=x_2_test, noise=noise)

        self.x_original = x
        self.y_original = y
        self.x = self.input_scaler.transform(x)
        self.y = self.output_scaler.transform(y).reshape(-1)
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

    def _get_x_y(self, x_1, x_2, noise):
        x = np.array(list(zip(x_1, x_2)))
        dataset_size = x.shape[0]
        y = x_1 + 0.3 * np.sin(2 * np.pi * (x_1 + np.random.normal(noise[0], noise[1], dataset_size))) + \
            0.3 * np.sin(4 * np.pi * (x_2 + np.random.normal(noise[0], noise[1], dataset_size))) + \
            np.random.normal(noise[0], noise[1], dataset_size)
        return x, y.reshape(-1, 1)

    def unnormalize_output(self, y_pred: torch.Tensor) -> torch.Tensor:
        y_pred = y_pred.numpy().reshape((-1, 1))
        y_pred_unnormalized = self.output_scaler.inverse_transform(y_pred).reshape(-1)
        return torch.Tensor(y_pred_unnormalized)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        x = torch.tensor(self.x[idx])
        y = torch.tensor(self.y[idx])
        return x, y
