import abc
import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

import gtimer as gt
from rlkit.core import logger, eval_util
from rlkit.core.rl_algorithm import BaseRLAlgorithm, _get_epoch_timings
from rlkit.core.dyna_rl_algorithm.torch_dyna_algorithm import \
        TorchDynaAlgorithm
from rlkit.data_management.replay_buffer import ReplayBuffer
from rlkit.samplers.data_collector import PathCollector
import rlkit.torch.pytorch_util as ptu


class OnlineDynaAlgorithm(TorchDynaAlgorithm, metaclass=abc.ABCMeta):

    def _train(self):
        self.training_mode(False)
        if self.min_num_steps_before_training > 0:
            self.expl_data_collector.collect_new_steps(
                self.max_path_length,
                self.min_num_steps_before_training,
                discard_incomplete_paths=False,
            )
            init_expl_paths = self.expl_data_collector.get_epoch_paths()
            self.replay_buffer.add_paths(init_expl_paths)
            self.expl_data_collector.end_epoch(-1)

            gt.stamp('initial exploration', unique=True)

        num_model_trains_per_step = (self.num_model_trains_per_loop
                // self.num_expl_steps_per_loop)
        num_model_steps_per_step = (self.num_model_steps_per_loop
                // self.num_expl_steps_per_loop)
        num_policy_trains_per_step = (self.num_policy_trains_per_loop
                // self.num_expl_steps_per_loop)
        for epoch in gt.timed_for(
                range(self._start_epoch, self.num_epochs),
                save_itrs=True,
        ):
            self.eval_data_collector.collect_new_paths(
                self.max_path_length,
                self.num_eval_steps_per_epoch,
                discard_incomplete_paths=True,
            )
            gt.stamp('evaluation sampling')

            iterator = range(self.num_expl_steps_per_loop)
            if not self.silent_inner_loop:
                iterator = tqdm(iterator)
            for _ in iterator:
                self.expl_data_collector.collect_new_steps(
                    self.max_path_length,
                    1,  # num steps
                    discard_incomplete_paths=False,
                )
                gt.stamp('exploration sampling', unique=False)

                # Do more training for the model.
                self._train_model(num_model_trains_per_step)
                # Do training on imaginary samples.
                self._train_agent(num_model_steps_per_step,
                        num_policy_trains_per_step)

            new_expl_paths = self.expl_data_collector.get_epoch_paths()
            self.replay_buffer.add_paths(new_expl_paths)
            gt.stamp('data storing', unique=False)

            self._end_epoch(epoch)
