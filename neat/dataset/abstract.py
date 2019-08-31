from torch.utils.data import Dataset


class NeatTestingDataset(Dataset):

    def generate_data(self):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, idx):
        raise NotImplementedError
