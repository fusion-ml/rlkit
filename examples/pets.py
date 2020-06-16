"""
An implementation of the Probabilistic Ensembles with Trajectory Sampling (PETS) algorithm
from Chua et al (2018).

"""
import sys
import gym
import torch

import rlkit.torch.pytorch_util as ptu
from rlkit.torch.PETS import Model, MPCPolicy, PETSTrainer
from rlkit.envs.wrappers import NormalizedBoxEnv
from rlkit.data_management.env_replay_buffer import EnvReplayBuffer
from rlkit.launchers.launcher_util import setup_logger
from rlkit.samplers.data_collector import MdpPathCollector
from rlkit.torch.torch_rl_algorithm import TorchBatchRLAlgorithm
from custom.mountain_car_continuous import mountain_car_continuous_reward
from custom.cartpole_swingup import CartPoleSwingUpEnv, cartpole_swingup_reward_v1

ptu.set_gpu_mode(True)


def experiment(variant):
    # expl_env = NormalizedBoxEnv(gym.make('BipedalWalker-v3'))
    # eval_env = NormalizedBoxEnv(gym.make('BipedalWalker-v3'))
    # expl_env = NormalizedBoxEnv(CartPoleSwingUpEnv())
    # eval_env = NormalizedBoxEnv(CartPoleSwingUpEnv())
    from custom.mjcartpole import CartpoleEnv, get_cp_reward # This import will fail if you don't have Mujoco.
    expl_env = CartpoleEnv()
    eval_env = CartpoleEnv()
    # expl_env = NormalizedBoxEnv(gym.make('MountainCarContinuous-v0'))
    # eval_env = NormalizedBoxEnv(gym.make('MountainCarContinuous-v0'))
    assert variant['policy']['num_particles'] % variant['model']['num_bootstrap'] == 0, "There must be an even number of particles per bootstrap"  # NOQA
    assert variant['algorithm_kwargs']['num_trains_per_train_loop'] % variant['model']['num_bootstrap'] == 0, "Must be an even number of train steps per bootstrap"  # NOQA
    obs_dim = expl_env.observation_space.low.size
    action_dim = expl_env.action_space.low.size

    model = Model(
                hidden_sizes=variant['model']['hidden_sizes'],
                obs_dim=obs_dim,
                action_dim=action_dim,
                num_bootstrap=variant['model']['num_bootstrap'],
                rew_function=get_cp_reward,
                # env=expl_env,
                # rew_function=mountain_car_continuous_reward  # for now
                )
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
    trainer = PETSTrainer(expl_env,
                          policy,
                          model,
                          lr=variant['lr'])
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
    torch.set_num_threads(8)
    variant = dict(
            policy=dict(
                num_particles=20,
                cem_horizon=25,
                cem_iters=5,
                cem_popsize=400,
                cem_num_elites=40,
                sampling_strategy='TSinf',
                optimizer='CEM',
                cem_alpha=0.1,
                opt_freq=1,
            ),
            algorithm_kwargs=dict(
                num_epochs=50,
                num_eval_steps_per_epoch=200,
                num_trains_per_train_loop=100000,
                num_expl_steps_per_train_loop=500,
                min_num_steps_before_training=10000,
                max_path_length=200,
                batch_size=256,
            ),
            # policy=dict(
            #     num_particles=10,
            #     cem_horizon=25,
            #     cem_iters=2,
            #     cem_popsize=10,
            #     cem_num_elites=3,
            #     sampling_strategy='TS1',
            #     optimizer='CEM',
            #     cem_alpha=0.1,
            #     opt_freq=25,
            # ),
            # algorithm_kwargs=dict(
            #     num_epochs=3000,
            #     num_eval_steps_per_epoch=400,
            #     num_trains_per_train_loop=100,
            #     num_expl_steps_per_train_loop=200,
            #     min_num_steps_before_training=200,
            #     max_path_length=200,
            #     batch_size=256,
            # ),
            model=dict(
                num_bootstrap=5,
                hidden_sizes=[500, 500, 500],
            ),
            replay_buffer_size=int(1e7),
            lr=0.001,
    )

    setup_logger(name, variant=variant)
    # import pudb; pudb.set_trace()
    experiment(variant)
