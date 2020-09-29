"""
Behavior clone trainer for discrete action environmnet.
"""
from collections import OrderedDict

import numpy as np
import torch
import torch.optim as optim
from torch import nn as nn

import rlkit.torch.pytorch_util as ptu
from rlkit.core.eval_util import create_stats_ordered_dict
from rlkit.torch.torch_rl_algorithm import TorchTrainer


class DiscreteBCTrainer(TorchTrainer):
    def __init__(
            self,
            classifier,
            learning_rate=1e-3,
    ):
        super().__init__()
        self.classifier = classifier
        self.learning_rate = learning_rate
        self.optimizer = optim.Adam(
            self.classifier.parameters(),
            lr=self.learning_rate,
        )
        self.criterion = nn.CrossEntropyLoss()
        self.eval_statistics = OrderedDict()
        self._n_train_steps_total = 0
        self._need_to_update_eval_statistics = True

    def train_from_torch(self, batch):
        obs = batch['observations']
        actions = torch.argmax(batch['actions'].long(), 1)
        net_out = self.classifier(obs)
        loss = self.criterion(net_out, actions)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        if self._need_to_update_eval_statistics:
            self._need_to_update_eval_statistics = False
            self.eval_statistics['CE Loss'] = np.mean(ptu.get_numpy(loss))

    def get_diagnostics(self):
        return self.eval_statistics

    def end_epoch(self, epoch):
        self._need_to_update_eval_statistics = True

    @property
    def networks(self):
        return [
            self.classifier,
        ]

    def get_snapshot(self):
        return dict(
            classifier=self.classifier,
        )
