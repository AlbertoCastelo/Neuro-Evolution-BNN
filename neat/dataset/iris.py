import random

import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
import os
from neat.dataset.abstract import NeatTestingDataset
import numpy as np


class IrisDataset(NeatTestingDataset):
    '''
    Dataset with 4 input variables and 3 classes
    '''

    def __init__(self, train_percentage, dataset_type='train', random_state=42, noise=0.0, label_noise=0.0, is_debug=False):
        directory = os.path.dirname(os.path.realpath(__file__))
        original_filename = ''.join([directory, f'/data/iris/iris.csv'])
        if noise > 0:
            filename_noisy = ''.join([directory, f'/data/iris/iris_{noise}.csv'])
            if os.path.isfile(filename_noisy):
                data = pd.read_csv(filename_noisy)
            else:
                # create noisy dataset and save it.
                data_original = pd.read_csv(original_filename)
                data = self._create_noisy_dataset(data_original, filename=filename_noisy, noise=noise)
        else:
            data = pd.read_csv(original_filename)
        self.data = data

        super().__init__(train_percentage=train_percentage, dataset_type=dataset_type,
                         random_state=random_state, noise=noise, label_noise=label_noise)

    def _generate_data(self):
        self.y = self.data['target'].values
        df_x = self.data.iloc[:, :4]
        self.x_original = df_x.values

        # self.x_original = self._add_noise(x=self.x_original)

        self.input_scaler = StandardScaler()
        self.input_scaler.fit(self.x_original)
        self.x = self.input_scaler.transform(self.x_original)

        self.x = torch.tensor(self.x).float()
        self.y = torch.tensor(self.y).long()

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        x = self.x[idx]
        y = self.y[idx]
        return x, y

    def _create_noisy_dataset(self, data, filename, noise):
        y = data['target'].values
        x = data.iloc[:, :4].values

        means = np.mean(x, 0)
        stds = np.std(x, 0)
        x_noisy = ((x - means) / stds +
                   np.random.normal(0, noise, size=x.shape)
                   + means) * stds

        data_noisy = pd.DataFrame(x_noisy, columns=data.columns[:4])
        data_noisy['target'] = y
        data_noisy.to_csv(filename, index=False)
        return data_noisy
