from torch.utils.data import Dataset


class NeatTestingDataset(Dataset):

    def generate_data(self):
        raise NotImplementedError
