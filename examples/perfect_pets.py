"""
An implementation of the Probabilistic Ensembles with Trajectory Sampling (PETS) algorithm
from Chua et al (2018). Don't learn a model, give the exact environment.

"""
import sys
import gym
import torch
import numpy as np

import rlkit.torch.pytorch_util as ptu
from rlkit.torch.PETS import Model, MPCPolicy, PETSTrainer
from rlkit.envs.wrappers import NormalizedBoxEnv
from rlkit.data_management.env_replay_buffer import EnvReplayBuffer
from rlkit.launchers.launcher_util import setup_logger
from rlkit.samplers.data_collector import MdpPathCollector
from rlkit.torch.torch_rl_algorithm import TorchBatchRLAlgorithm, TorchTrainer
from rlkit.torch.core import torch_ify
from custom.mountain_car_continuous import mountain_car_continuous_reward
from custom.cartpole_swingup import CartPoleSwingUpEnv, cartpole_swingup_reward_v1

ptu.set_gpu_mode(False)

class CheatModel():


    def __init__(
            self,
            env,
            env_name,
            rew_function=None,
    ):
        self.env = env
        self.env_name = env_name
        self.rew_function = rew_function
        self.lb = torch_ify(self.env._wrapped_env.action_space.low)
        self.ub = torch_ify(self.env._wrapped_env.action_space.high)
        self.trained_at_all = True


    def forward(
            self,
            obs,
            action,
            network_idx=None,
            return_net_outputs=False,
    ):
        self._set_env_state(obs)
        nxt, rew, done, _ = self.env.step(action)
        return nxt, rew

    def denormalize_action(self, action):
        assert (torch.abs(action) <= 1).all()
        scaled_action = self.lb + (action + 1.) * 0.5 * (self.ub - self.lb)
        # scaled_action = torch.clamp(scaled_action, self.lb, self.ub)
        return scaled_action

    def unroll(self, obs, action_sequence, sampling_strategy):
        obs = np.asarray(obs)
        actions = np.asarray(action_sequence)
        nxts = []
        rews = []
        for bidx in range(obs.shape[0]):
            batch_nxts = []
            batch_rews = []
            bobs = obs[bidx]
            self.env.reset()
            self._set_env_state(bobs)
            for aidx in range(actions.shape[1]):
                nxt, rew, done, _ = self.env.step(actions[bidx, aidx])
                batch_nxts.append(nxt)
                batch_rews.append(rew)
            nxts.append(batch_nxts)
            rews.append(batch_rews)
        return np.asarray(nxts), np.asarray(rews)

    def bound_loss(self):
        return 0

    def _set_env_state(self, obs):
        if self.env_name == 'CartPoleSwingUp':
            state = np.asarray([obs[0], obs[1], np.arccos(obs[2]), obs[4]])
            self.env._wrapped_env.state = state
        elif self.env_name == 'Cartpole':
            self.env._wrapped_env.set_state(obs[:2], obs[2:])
        else:
            self.env._wrapped_env.env.state = obs


class MockTrainer(TorchTrainer):

    def train_from_torch(self, batch):
        pass

    @property
    def networks(self):
        return []

    def get_snapshot(self):
        return {}

    def get_diagnostics(self):
        return {}

    def end_epoch(self, epoch):
        pass

def experiment(variant):
    env_name = 'Cartpole'
    if env_name == 'CartPoleSwingUp':
        expl_env = NormalizedBoxEnv(CartPoleSwingUpEnv())
        eval_env = NormalizedBoxEnv(CartPoleSwingUpEnv())
        model_env = NormalizedBoxEnv(CartPoleSwingUpEnv())
    elif env_name == 'Cartpole':
        from custom.mjcartpole import CartpoleEnv
        expl_env = NormalizedBoxEnv(CartpoleEnv())
        eval_env = NormalizedBoxEnv(CartpoleEnv())
        model_env = NormalizedBoxEnv(CartpoleEnv())
    else:
        expl_env = NormalizedBoxEnv(gym.make(env_name))
        eval_env = NormalizedBoxEnv(gym.make(env_name))
        model_env = NormalizedBoxEnv(gym.make(env_name))
    # expl_env = NormalizedBoxEnv(gym.make('BipedalWalker-v3'))
    # eval_env = NormalizedBoxEnv(gym.make('BipedalWalker-v3'))
    # expl_env = NormalizedBoxEnv(gym.make('MountainCarContinuous-v0'))
    # eval_env = NormalizedBoxEnv(gym.make('MountainCarContinuous-v0'))
    assert variant['policy']['num_particles'] % variant['model']['num_bootstrap'] == 0, "There must be an even number of particles per bootstrap"  # NOQA
    assert variant['algorithm_kwargs']['num_trains_per_train_loop'] % variant['model']['num_bootstrap'] == 0, "Must be an even number of train steps per bootstrap"  # NOQA
    obs_dim = expl_env.observation_space.low.size
    action_dim = expl_env.action_space.low.size

    model = CheatModel(model_env, env_name)
    policy = MPCPolicy(
            model=model,
            obs_dim=obs_dim,
            action_dim=action_dim,
            num_particles=variant['policy']['num_particles'],
            cem_horizon=variant['policy']['cem_horizon'],
            cem_iters=variant['policy']['cem_iters'],
            cem_popsize=variant['policy']['cem_popsize'],
            cem_num_elites=variant['policy']['cem_num_elites'],
            sampling_strategy=variant['policy']['sampling_strategy'],
            optimizer=variant['policy']['optimizer'],
            opt_freq=variant['policy']['opt_freq'],
            cem_alpha=variant['policy']['cem_alpha'],
            )
    trainer = MockTrainer()
    eval_path_collector = MdpPathCollector(
            eval_env,
            policy,
    )
    expl_path_collector = MdpPathCollector(
            expl_env,
            policy,
    )
    replay_buffer = EnvReplayBuffer(
        variant['replay_buffer_size'],
        expl_env,
    )
    algorithm = TorchBatchRLAlgorithm(
            trainer=trainer,
            exploration_env=expl_env,
            evaluation_env=eval_env,
            exploration_data_collector=expl_path_collector,
            evaluation_data_collector=eval_path_collector,
            replay_buffer=replay_buffer,
            **variant['algorithm_kwargs']
    )
    algorithm.to(ptu.device)
    algorithm.train()


if __name__ == '__main__':
    name = sys.argv[1]
    variant = dict(
            policy=dict(
                num_particles=1,
                cem_horizon=25,
                cem_iters=5,
                cem_popsize=400,
                cem_num_elites=40,
                sampling_strategy='TS1',
                optimizer='CEM',
                opt_freq=1,
                cem_alpha=0.1
            ),
            model=dict(
                num_bootstrap=1,
                hidden_sizes=[500, 500, 500],
            ),
            replay_buffer_size=int(1e7),
            algorithm_kwargs=dict(
                num_epochs=3000,
                num_eval_steps_per_epoch=200,
                num_trains_per_train_loop=0,
                num_expl_steps_per_train_loop=10,
                min_num_steps_before_training=0,
                max_path_length=200,
                batch_size=256,
            ),
            lr=0.001,
    )

    setup_logger(name, variant=variant)
    # import pudb; pudb.set_trace()
    experiment(variant)
