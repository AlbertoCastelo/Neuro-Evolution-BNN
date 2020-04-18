import os
from glob import glob
from sklearn.model_selection import train_test_split
import torch
import pandas as pd
import numpy as np
from torchvision.transforms import transforms
from PIL import Image

from neat.dataset.abstract import NeatTestingDataset
MEAN_1CHANNEL = (0.6068, )
STD_1CHANNEL = (0.2563, )


class HistoPathologicCancer(NeatTestingDataset):
    """Histo-Pathologic Cancer Dataset with binary labeling: Finding/No-Finding"""

    def __init__(self, train_percentage, dataset_type, random_state=42, path=None, img_size=64,
                 transform=None, n_channels=1, is_debug=True):
        self.path = path if path is not None \
            else '/home/alberto/Desktop/datasets/histopathologic-cancer-detection/'
        self.img_size = img_size

        self.n_channels = n_channels
        self.is_debug = is_debug

        # define transformation
        self.transform = transform
        if self.transform is None:
            self.transform = transforms.Compose([
                                transforms.CenterCrop(32),
                                # transforms.
                                # transforms.Resize((img_size, img_size)),
                                transforms.ToTensor(),
                                transforms.Normalize(MEAN_1CHANNEL, STD_1CHANNEL)])

        self.transform_target = transforms.ToTensor()

        self.df_data = pd.read_csv(os.path.join(self.path, 'train_labels.csv'))

        if is_debug:
            self.df_data = self.df_data[:20000]

        super().__init__(train_percentage=train_percentage, dataset_type=dataset_type,
                         random_state=random_state)

    def generate_data(self):
        def _data_generator(self):
            for i in range(len(self)):
                img, _ = self[i]
                # import matplotlib.pyplot as plt
                # plt.imshow(img.squeeze(0).numpy())
                # plt.show()
                yield img.reshape(1, -1)

        self.data = torch.cat(tuple(_data_generator(self=self)), 0)
        self.targets = torch.tensor(self.df_data['label'].values).long()

        self.x = self.data
        self.y = self.targets

        self._generate_train_test_sets()

    def __len__(self):
        return len(self.df_data)

    def __getitem__(self, idx):
        item = self.df_data.iloc[idx]
        path = os.path.join(self.path, 'train', ''.join([item['id'], '.tif']))
        if self.n_channels == 1:
            img = self.get_grayscale_image_from_file(path)
        elif self.n_channels == 3:
            img = self.get_rgb_image_from_file(path)
        else:
            raise Exception

        # img = self.get_rgb_image_from_file()
        img = self.transform(img)
        label = np.array(item['label'])
        label = torch.from_numpy(label)
        return img, label

    def load_image_from_file(self, filename):
        return Image.open(filename)

    def get_rgb_image_from_file(self, filename):
        return self.load_image_from_file(filename).convert('RGB')

    def get_grayscale_image_from_file(self, filename):
        return self.load_image_from_file(filename).convert('L')