from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
import torch


class NeatTestingDataset(Dataset):
    def __init__(self, train_percentage, dataset_type, random_state):
        self.train_percentage = train_percentage
        self.dataset_type = dataset_type
        self.random_state = random_state

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

    def _get_data_limit(self):
        return int(round(len(self.x) * self.train_percentage))

    def generate_data(self):
        raise NotImplementedError
