import abc
import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

import gtimer as gt
from rlkit.core import logger, eval_util
from rlkit.core.rl_algorithm import BaseRLAlgorithm, _get_epoch_timings
from rlkit.data_management.replay_buffer import ReplayBuffer
from rlkit.samplers.data_collector import PathCollector
import rlkit.torch.pytorch_util as ptu


class TorchDynaRLAlgorithm(BaseRLAlgorithm, metaclass=abc.ABCMeta):

    def __init__(
            self,
            agent_trainer,
            model_trainer,
            exploration_env,
            evaluation_env,
            model_env,
            exploration_data_collector: PathCollector,
            evaluation_data_collector: PathCollector,
            model_data_collector: PathCollector,
            replay_buffer: ReplayBuffer,
            imaginary_replay_buffer: ReplayBuffer,
            batch_size,
            max_path_length,
            max_model_path_length,
            num_epochs,
            num_eval_steps_per_epoch,
            num_expl_steps_per_loop,
            num_model_steps_per_loop,
            num_policy_trains_per_loop,
            num_model_trains_per_loop,
            num_policy_loops_per_epoch,
            min_num_steps_before_training=0,
            silent_inner_loop=True,
    ):
        # Set parameters from in parent constructor.
        self.post_epoch_funcs = []
        self._start_epoch = 0
        # Set class members.
        self.trainer = agent_trainer
        self.model_trainer = model_trainer
        self.expl_env = exploration_env
        self.eval_env = evaluation_env
        self.model_env = model_env
        self.expl_data_collector = exploration_data_collector
        self.eval_data_collector = evaluation_data_collector
        self.model_data_collector = model_data_collector
        self.replay_buffer = replay_buffer
        self.imaginary_replay_buffer = imaginary_replay_buffer
        self.batch_size = batch_size
        self.max_path_length = max_path_length
        self.max_model_path_length = max_model_path_length
        self.num_epochs = num_epochs
        self.num_eval_steps_per_epoch = num_eval_steps_per_epoch
        self.num_policy_trains_per_loop = num_policy_trains_per_loop
        self.num_model_trains_per_loop = num_model_trains_per_loop
        self.num_expl_steps_per_loop = num_expl_steps_per_loop
        self.num_model_steps_per_loop = num_model_steps_per_loop
        self.num_policy_loops_per_epoch = num_policy_loops_per_epoch
        self.min_num_steps_before_training = min_num_steps_before_training
        self.silent_inner_loop = silent_inner_loop
        self.action_dim = exploration_env.action_space.low.size

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
            self._train_model()
            # Do training on imaginary samples.
            self._train_agent()
            self._end_epoch(epoch)

    def _train_model(self):
        tr_x, tr_y = self.replay_buffer.get_transition_data()
        # TODO: Make validation set maybe?
        data_loader = DataLoader(
                TensorDataset(torch.Tensor(tr_x), torch.Tensor(tr_y)),
                batch_size=self.batch_size,
                pin_memory=True,
                shuffle=False,
        )
        self.training_mode(True)
        for _ in range(self.num_model_trains_per_loop):
            for xi, yi in data_loader:
                xi = xi.to(ptu.device)
                yi = yi.to(ptu.device)
                self.model_trainer.train_from_torch({
                    'observations': xi[:, :-self.action_dim],
                    'actions': xi[:, -self.action_dim:],
                    'next_observations': yi,
                })
        gt.stamp('model training', unique=False)
        self.training_mode(False)

    def _train_agent(self):
        iterator = range(self.num_policy_loops_per_epoch)
        if not self.silent_inner_loop:
            iterator = tqdm(iterator)
        for inner_ep in iterator:
            new_img_paths = self.model_data_collector.collect_new_paths(
                    self.max_model_path_length,
                    self.num_model_steps_per_loop,
                    discard_incomplete_paths=False,
            )
            gt.stamp('model rollout', unique=False)
            self.imaginary_replay_buffer.add_paths(new_img_paths)
            gt.stamp('imaginary data storing', unique=False)
            self.training_mode(True)
            for _ in range(self.num_policy_trains_per_loop):
                train_data = self.imaginary_replay_buffer.random_batch(
                    self.batch_size)
                self.trainer.train(train_data)
            gt.stamp('agent training', unique=False)
            self.training_mode(False)

    def _end_epoch(self, epoch):
        super(TorchDynaRLAlgorithm, self)._end_epoch(epoch)
        self.model_data_collector.end_epoch(epoch)
        self.model_trainer.end_epoch(epoch)
        self.imaginary_replay_buffer.end_epoch(epoch)

    def _get_snapshot(self):
        snapshot = super(TorchDynaRLAlgorithm, self)._get_snapshot()
        for k, v in self.model_trainer.get_snapshot().items():
            snapshot['model/' + k] = v
        for k, v in self.model_data_collector.get_snapshot().items():
            snapshot['imaginary/' + k] = v
        for k, v in self.imaginary_replay_buffer.get_snapshot().items():
            snapshot['imaginary_buffer/' + k] = v
        return snapshot

    def _log_stats(self, epoch):
        logger.log("Epoch {} finished".format(epoch), with_timestamp=True)

        """
        Replay Buffer
        """
        logger.record_dict(
            self.replay_buffer.get_diagnostics(),
            prefix='replay_buffer/'
        )
        logger.record_dict(
            self.imaginary_replay_buffer.get_diagnostics(),
            prefix='imaginary_buffer/'
        )

        """
        Agent Trainer
        """
        logger.record_dict(self.trainer.get_diagnostics(), prefix='trainer/')

        """
        Model Trainer
        """
        logger.record_dict(self.model_trainer.get_diagnostics(), prefix='model/')

        """
        Exploration
        """
        logger.record_dict(
            self.expl_data_collector.get_diagnostics(),
            prefix='exploration/'
        )
        expl_paths = self.expl_data_collector.get_epoch_paths()
        if hasattr(self.expl_env, 'get_diagnostics'):
            logger.record_dict(
                self.expl_env.get_diagnostics(expl_paths),
                prefix='exploration/',
            )
        logger.record_dict(
            eval_util.get_generic_path_information(expl_paths),
            prefix="exploration/",
        )
        """
        Evaluation
        """
        logger.record_dict(
            self.eval_data_collector.get_diagnostics(),
            prefix='evaluation/',
        )
        eval_paths = self.eval_data_collector.get_epoch_paths()
        if hasattr(self.eval_env, 'get_diagnostics'):
            logger.record_dict(
                self.eval_env.get_diagnostics(eval_paths),
                prefix='evaluation/',
            )
        logger.record_dict(
            eval_util.get_generic_path_information(eval_paths),
            prefix="evaluation/",
        )
        """
        Imaginary
        """
        logger.record_dict(
            self.model_data_collector.get_diagnostics(),
            prefix='imaginary/',
        )
        img_paths = self.model_data_collector.get_epoch_paths()
        if hasattr(self.model_env, 'get_diagnostics'):
            logger.record_dict(
                self.model_env.get_diagnostics(img_paths),
                prefix='imaginary/',
            )
        logger.record_dict(
            eval_util.get_generic_path_information(img_paths),
            prefix="imaginary/",
        )

        """
        Misc
        """
        gt.stamp('logging')
        logger.record_dict(_get_epoch_timings())
        logger.record_tabular('Epoch', epoch)
        logger.dump_tabular(with_prefix=False, with_timestamp=False)

    def to(self, device):
        for net in self.trainer.networks:
            net.to(device)
        for net in self.model_trainer.networks:
            net.to(device)
        self.model_trainer.model.max_logvar.to(device)
        self.model_trainer.model.min_logvar.to(device)

    def training_mode(self, mode):
        for net in self.trainer.networks:
            net.train(mode)
        for net in self.model_trainer.networks:
            net.train(mode)
