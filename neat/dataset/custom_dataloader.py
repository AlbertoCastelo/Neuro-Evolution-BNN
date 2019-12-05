from torch.utils.data import Dataset, DataLoader

from neat.configuration import get_configuration


class CustomDataLoader:
    def __init__(self, dataset: Dataset, shuffle=True):
        self.dataset = dataset
        self.shuffle = shuffle

        self.iterator = self._get_iterator()

    def _get_iterator(self):
        return self.dataset.x, self.dataset.y

    def __iter__(self):
        return self.iterator

    def __len__(self):
        return len(self.dataset)


def get_data_loader(dataset: Dataset, batch_size=None):
    config = get_configuration()
    parallel_evaluation = config.parallel_evaluation
    if not parallel_evaluation:
        return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=1)
    else:
        return CustomDataLoader(dataset, shuffle=True)
