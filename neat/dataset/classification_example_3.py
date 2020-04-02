import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler, LabelBinarizer, MultiLabelBinarizer
from torch.utils.data import Dataset
import numpy as np

from neat.dataset.abstract import NeatTestingDataset


class ClassificationExample2Dataset(NeatTestingDataset):
    '''
    Dataset with 2 input variables and 2 classes
    '''
    SIZE = 5000
    X1_MIN = -1.0
    X1_MAX = 1.0
    X2_MIN = -1.0
    X2_MAX = 1.0

    def __init__(self, train_percentage, dataset_type='train', is_debug=False):
        if dataset_type not in ['train', 'validation', 'test']:
            raise ValueError(f'Dataset Type {dataset_type} is not valid')
        self.dataset_type = dataset_type
        self.is_debug = is_debug

        self.input_scaler = None
        self.output_transformer = None

        self.x_original = None
        self.y_original = None
        self.x = None
        self.y = None
        super().__init__(train_percentage=train_percentage, dataset_type=dataset_type)

    def generate_data(self):
        self.input_scaler = StandardScaler()
        self.output_transformer = LabelBinarizer()

        x1 = np.random.uniform(self.X1_MIN, self.X1_MAX, size=(self.SIZE, 1))
        x2 = np.random.uniform(self.X2_MIN, self.X2_MAX, size=(self.SIZE, 1))
        x_train = np.concatenate((x1, x2), axis=1)
        x_train, y_train = self._get_x_y(x=x_train)

        self.input_scaler.fit(x_train)
        self.output_transformer.fit(y_train)

        if self.dataset_type == 'train':
            x = x_train
            y = y_train
        elif self.dataset_type == 'test':
            x1 = np.random.uniform(self.X1_MIN, self.X1_MAX, size=(self.SIZE, 1))
            x2 = np.random.uniform(self.X2_MIN, self.X2_MAX, size=(self.SIZE, 1))
            x_test = np.concatenate((x1, x2), axis=1)
            x, y = self._get_x_y(x=x_test)

        self.x_original = x
        self.y_original = y

        self.x = self.input_scaler.transform(x)
        self.y = self.output_transformer.transform(y).squeeze()

        if self.is_debug:
            self.x = x[:512]
            self.y = y[:512]

        self.x = torch.tensor(self.x).float()
        self.y = torch.tensor(self.y).long()

        data_limit = self._get_data_limit()
        self.x_train = self.x[:data_limit]
        self.y_train = self.y[:data_limit]

        self.x_test = self.x[data_limit:]
        self.y_test = self.y[data_limit:]

    def get_separation_line(self):
        x1 = np.linspace(self.X1_MIN, self.X1_MAX, 200)
        x2 = self._get_x2_limit(x1)
        return x1, x2

    def _get_x_y(self, x):
        independent_cols = ['x1', 'x2']
        dependent_col = 'y'
        df_grid = pd.DataFrame(x, columns=independent_cols)

        def _get_class(x):
            x1 = x.x1
            x2 = x.x2

            if x2 <= -0.3:
                return 'A'
            elif x2 > -0.3 and x2 <= 0.3:
                return 'B'
            return 'C'

        df_grid[dependent_col] = df_grid.apply(lambda x: _get_class(x), axis=1)
        x = df_grid.loc[:, independent_cols].to_numpy()
        y = df_grid.loc[:, dependent_col].to_numpy()
        return x, y

    def _get_x2_limit(self, x1):
        x2_limit = x1 + 0.3 * np.sin(2 * np.pi * x1)  + \
              0.3 * np.sin(4 * np.pi * x1 )
        return x2_limit

    def unnormalize_output(self, y_pred: torch.Tensor) -> torch.Tensor:
        y_pred = y_pred.numpy().reshape((-1, 1))
        y_pred_unnormalized = self.output_transformer.inverse_transform(y_pred).reshape(-1)
        return torch.Tensor(y_pred_unnormalized)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        x = self.x[idx]
        y = self.y[idx]
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
        # self.x = x
        # self.y = y.reshape((-1, 1))
        self.x = self.input_scaler.transform(x)
        self.y = self.output_scaler.transform(y).reshape((-1, 1))

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
