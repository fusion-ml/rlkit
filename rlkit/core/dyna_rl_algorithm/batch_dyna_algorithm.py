import abc
import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import numpy as np

import gtimer as gt
from rlkit.core import logger, eval_util
from rlkit.core.rl_algorithm import BaseRLAlgorithm, _get_epoch_timings
from rlkit.core.dyna_rl_algorithm.torch_dyna_algorithm import \
        TorchDynaAlgorithm
from rlkit.data_management.replay_buffer import ReplayBuffer
from rlkit.samplers.data_collector import PathCollector
import rlkit.torch.pytorch_util as ptu


class BatchDynaAlgorithm(TorchDynaAlgorithm, metaclass=abc.ABCMeta):

    def _train(self):
        # Collect initial data to train model on.
        if self.min_num_steps_before_training > 0:
            init_expl_paths = self.expl_data_collector.collect_new_paths(
                self.max_path_length,
                self.min_num_steps_before_training,
                discard_incomplete_paths=False,
            )
            self.replay_buffer.add_paths(init_expl_paths)
            self.expl_data_collector.end_epoch(-1)
        for epoch in gt.timed_for(
                range(self._start_epoch, self.num_epochs),
                save_itrs=True,
        ):
            # Do evaluation steps.
            self.eval_data_collector.collect_new_paths(
                self.max_path_length,
                self.num_eval_steps_per_epoch,
                discard_incomplete_paths=True,
            )
            gt.stamp('evaluation sampling')
            # Collect more exploration samples from environment.
            new_expl_paths = self.expl_data_collector.collect_new_paths(
                self.max_path_length,
                self.num_expl_steps_per_loop,
                discard_incomplete_paths=False,
            )
            gt.stamp('exploration sampling', unique=False)
            self.replay_buffer.add_paths(new_expl_paths)
            gt.stamp('data storing', unique=False)
            # Do more training for the model.
            self._train_model(self.num_model_trains_per_loop)
            # Do training on imaginary samples.
            num_paths = int(np.ceil(self.num_model_steps_per_loop \
                    / self.max_model_path_length))
            iterator = range(num_paths)
            if not self.silent_inner_loop:
                iterator = tqdm(iterator)
            num_steps_taken = 0
            for _ in iterator:
                steps_to_take = min([
                        self.max_model_path_length,
                        self.num_model_steps_per_loop - num_steps_taken
                ])
                self._train_agent(
                        steps_to_take,
                        self.num_policy_trains_per_loop,
                    )
                num_steps_taken += steps_to_take
            self._end_epoch(epoch)
