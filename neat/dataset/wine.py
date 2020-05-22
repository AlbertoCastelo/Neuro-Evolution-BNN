import pandas as pd
import torch
from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler
import os
from neat.dataset.abstract import NeatTestingDataset
import numpy as np


class WineDataset(NeatTestingDataset):
    '''
    Dataset with 4 input variables and 3 classes
    '''

    def __init__(self, train_percentage, dataset_type='train', random_state=42, noise=0.0, label_noise=0.0):
        # directory = os.path.dirname(os.path.realpath(__file__))
        if noise > 0:
            raise NotImplementedError
        x, y = load_wine(return_X_y=True)
        data = pd.DataFrame(x)
        data['target'] = y
        self.data = data

        super().__init__(train_percentage=train_percentage, dataset_type=dataset_type,
                         random_state=random_state, noise=noise, label_noise=label_noise)

    def _generate_data(self):
        self.y = self.data['target'].values
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
