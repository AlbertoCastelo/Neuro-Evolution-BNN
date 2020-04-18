import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler, LabelBinarizer, MultiLabelBinarizer
import os
from neat.dataset.abstract import NeatTestingDataset


class IrisDataset(NeatTestingDataset):
    '''
    Dataset with 4 input variables and 3 classes
    '''

    def __init__(self, train_percentage, dataset_type='train', random_state=42, is_debug=False):
        filename = ''.join([os.path.dirname(os.path.realpath(__file__)), '/data/iris/iris.csv'])
        self.data = pd.read_csv(filename)
        # self.data = pd.read_csv('/home/alberto/Desktop/repos/master_thesis/Neuro-Evolution-BNN/'
        #                         'neat/dataset/data/iris/iris.csv')

        super().__init__(train_percentage=train_percentage, dataset_type=dataset_type,
                         random_state=random_state)

    def generate_data(self):
        self.y = self.data['target'].values
        df_x = self.data.iloc[:, :4]
        self.x_original = df_x.values

        # self._add_noise(x_original)

        self.input_scaler = StandardScaler()
        self.input_scaler.fit(self.x_original)
        self.x = self.input_scaler.transform(self.x_original)


        self.x = torch.tensor(self.x).float()
        self.y = torch.tensor(self.y).long()

        self._generate_train_test_sets()

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        x = self.x[idx]
        y = self.y[idx]
        return x, y

    def _add_noise(self, x_original):
        pass