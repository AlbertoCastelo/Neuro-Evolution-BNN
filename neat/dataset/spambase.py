import random

import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
import os
from neat.dataset.abstract import NeatTestingDataset
import numpy as np


class SpamBaseDataset(NeatTestingDataset):
    '''
    Dataset with 4 input variables and 3 classes
    '''

    def __init__(self, train_percentage, dataset_type='train', random_state=42, noise=0.0, label_noise=0.0, is_debug=False):
        directory = os.path.dirname(os.path.realpath(__file__))
        original_filename = ''.join([directory, f'/data/spambase/spambase.data'])
        if noise > 0:
            raise NotImplementedError
        else:
            data = self._read_data(filename=original_filename)
        self.data = data

        super().__init__(train_percentage=train_percentage, dataset_type=dataset_type,
                         random_state=random_state, noise=noise, label_noise=label_noise)

    def _generate_data(self):
        self.y = self.data[57].values
        df_x = self.data.iloc[:, :-1]
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

    def _read_data(self, filename):
        with open(filename) as f:
            data = f.read()
        lines = data.split('\n')
        rows = []
        for line in lines[:-1]:
            rows.append(list(map(float, line.split(','))))
        data = pd.DataFrame(rows)
        return data
