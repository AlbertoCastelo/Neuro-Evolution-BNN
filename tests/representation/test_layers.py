from unittest import TestCase

import torch

from neat.representation.layers import StochasticLinearParameters, StochasticLinear


class TestStochasticLayer(TestCase):

    def setUp(self) -> None:
        self.in_features = 3
        self.out_features = 2

        qw_mean = torch.zeros((self.out_features, self.in_features))
        qw_mean[0][1] = 3.0
        qw_mean[1][0] = 1.0
        qw_mean[1][2] = 2.0
        qw_logvar = torch.ones((self.out_features, self.in_features)) * (-4.0)
        qb_mean = torch.zeros(self.out_features)
        qb_logvar = torch.ones(self.out_features) * (-4.0)

        self.parameters = StochasticLinearParameters.create(qw_mean=qw_mean,
                                                            qw_logvar=qw_logvar,
                                                            qb_mean=qb_mean,
                                                            qb_logvar=qb_logvar)
        self.batch_size = 1
        self.input = torch.zeros((self.batch_size, self.in_features))

    def test_forward_return_correct_shapes(self):
        layer = StochasticLinear(in_features=self.in_features,
                                 out_features=self.out_features,
                                 parameters=self.parameters)

        output, kl = layer(self.input)

        self.assertEqual(type(output), torch.Tensor)
        self.assertEqual(type(kl), torch.Tensor)
        self.assertEqual(kl.shape, torch.Size([]))
