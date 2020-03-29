from torch.utils.data import Dataset


class NeatTestingDataset(Dataset):
    def __init__(self, train_percentage, dataset_type):
        self.train_percentage = train_percentage
        self.dataset_type = dataset_type
        self.train = False
        self.x = None
        self.y = None
        self.x_train = None
        self.y_train = None
        self.x_test = None
        self.y_test = None
        if dataset_type == 'train':
            self.train = True

    def _get_data_limit(self):
        return int(round(len(self.x) * self.train_percentage))

    def generate_data(self):
        raise NotImplementedError
