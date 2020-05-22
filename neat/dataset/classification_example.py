import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler, LabelBinarizer
import numpy as np

from neat.dataset.abstract import NeatTestingDataset


class ClassificationExample1Dataset(NeatTestingDataset):
    '''
    Dataset with 2 input variables and 2 classes
    '''
    SIZE = 5000
    X1_MIN = -1.0
    X1_MAX = 1.0
    X2_MIN = -1.0
    X2_MAX = 1.0

    def __init__(self, train_percentage, dataset_type='train', random_state=42, noise=0.0, label_noise=0.0,
                 is_debug=False):
        if dataset_type not in ['train', 'validation', 'test']:
            raise ValueError(f'Dataset Type {dataset_type} is not valid')
        self.dataset_type = dataset_type
        self.is_debug = is_debug

        self.input_scaler = None
        self.output_transformer = None

        self.x_original = None
        self.y_original = None
        super().__init__(train_percentage=train_percentage, dataset_type=dataset_type,
                         random_state=random_state, noise=noise, label_noise=label_noise)

    def _generate_data(self):
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

        # self._generate_train_test_sets()

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
            limit = self._get_x2_limit(x1=x1)
            if x2 >= limit:
                return 'B'
            return 'A'
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
