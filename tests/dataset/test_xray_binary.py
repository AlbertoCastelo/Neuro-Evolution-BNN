from unittest import TestCase

import torch

from neat.dataset.classification_xray_binary import XRayBinary


class TestXRayBinary(TestCase):
    def test_generate_data(self):
        dataset = XRayBinary(train_percentage=0.5, dataset_type='test', is_debug=True)
        dataset.generate_data()

        x = dataset.x
        y = dataset.y
        print(len(dataset))
        print(y.sum())

        self.assertEqual(type(x), torch.Tensor)
        self.assertEqual(type(y), torch.Tensor)
        self.assertEqual(len(x), len(y))
        self.assertEqual(y.shape, (len(y),))
