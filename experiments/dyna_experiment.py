"""
Experiments for Dyna style MBRL.
"""
import argparse
from copy import deepcopy
import sys
import os
import gym
import torch
import numpy as np

import rlkit.torch.pytorch_util as ptu
from rlkit.torch.PETS import Model, MPCPolicy, PETSTrainer
from rlkit.envs.wrappers import NormalizedBoxEnv, ModelEnv
from rlkit.data_management.env_replay_buffer import EnvReplayBuffer
from rlkit.launchers.launcher_util import setup_logger
from rlkit.samplers.data_collector import MdpPathCollector
from rlkit.torch.dyna_rl_algorithm import TorchDynaRLAlgorithm
from rlkit.torch.dyn_model_trainer import DynModelTrainer
from rlkit.torch.networks import FlattenMlp
from rlkit.torch.sac.policies import TanhGaussianPolicy, MakeDeterministic
from rlkit.torch.sac.sac import SACTrainer

DEFAULT_VARIANT = dict(
    run_id='',
    seed=0,
    cuda_device='',
    environment='Cartpole',
    dyna_variant='standard', # Can also be mbpo.
    algorithm='SAC',
    agent_layer_size=256,
    replay_buffer_size=int(1E6),
    imaginary_replay_size=int(5E5),
    # algorithm_kwargs=dict(
    #     num_epochs=50,
    #     num_eval_steps_per_epoch=1000,
    #     num_expl_steps_per_loop=1000,
    #     num_model_steps_per_loop=1000,
    #     num_model_trains_per_loop=200,
    #     num_policy_trains_per_loop=200,
    #     num_policy_loops_per_epoch=100,
    #     min_num_steps_before_training=10000,
    #     max_path_length=200,
    #     max_model_path_length=200,
    #     batch_size=1024,
    #     silent_inner_loop=False,
    # ),
    algorithm_kwargs=dict(
        num_epochs=5,
        num_eval_steps_per_epoch=200,
        num_expl_steps_per_loop=5,
        num_model_steps_per_loop=5,
        num_model_trains_per_loop=1,
        num_policy_trains_per_loop=5,
        num_policy_loops_per_epoch=500,
        min_num_steps_before_training=5,
        max_path_length=200,
        max_model_path_length=200,
        batch_size=256,
        silent_inner_loop=False,
    ),
    agent_trainer_kwargs=dict(
        discount=0.99,
        soft_target_tau=5e-3,
        target_update_period=1,
        policy_lr=3E-4,
        qf_lr=3E-4,
        reward_scale=1,
        use_automatic_entropy_tuning=True,
    ),
    model_trainer_kwargs=dict(
        lr=1E-3,
    ),
    model_kwargs=dict(
        hidden_sizes=[500, 500, 500],
        num_bootstrap=5,
    ),
)

def launch_variant(variant):
    # Set logging, device, and seed.
    if variant['run_id'] != '':
        setup_logger(variant['run_id'], variant=variant)
    if variant['cuda_device'] != '':
        os.environ['CUDA_VISIBLE_DEVICES'] = variant['cuda_device']
        ptu.set_gpu_mode(True)
    else:
        ptu.set_gpu_mode(False)
    if not variant['seed'] is None:
        torch.manual_seed(variant['seed'])
        np.random.seed(variant['seed'])
    # Get environment.
    if variant['environment'] == 'Cartpole':
        from examples.custom.mjcartpole import CartpoleEnv, np_get_cp_reward
        expl_env = NormalizedBoxEnv(CartpoleEnv())
        eval_env = NormalizedBoxEnv(CartpoleEnv())
        true_env = CartpoleEnv()
        reward_func = np_get_cp_reward
        horizon = 200
    else:
        raise ValueError('Unknown environment %s.' % variant['environment'])
    obs_dim = expl_env.observation_space.low.size
    action_dim = expl_env.action_space.low.size
    # Set up replay buffers.
    replay_buffer = EnvReplayBuffer(
        variant['replay_buffer_size'],
        expl_env,
    )
    imaginary_replay_buffer = EnvReplayBuffer(
        variant['imaginary_replay_size'],
        expl_env,
    )
    # Set Up model.
    model = Model(
            obs_dim=obs_dim,
            action_dim=action_dim,
            **variant['model_kwargs'],
    )
    model_env = ModelEnv(
            true_env,
            model,
            reward_func=reward_func,
            horizon=horizon,
            replay_buffer=replay_buffer if variant['dyna_variant'] == 'mbpo' else None,
    )
    model_trainer = DynModelTrainer(
            model,
            **variant['model_trainer_kwargs'],
    )
    # Set up policy.
    M = variant['agent_layer_size']
    qf1 = FlattenMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        hidden_sizes=[M, M],
    )
    qf2 = FlattenMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        hidden_sizes=[M, M],
    )
    target_qf1 = FlattenMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        hidden_sizes=[M, M],
    )
    target_qf2 = FlattenMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        hidden_sizes=[M, M],
    )
    policy = TanhGaussianPolicy(
        obs_dim=obs_dim,
        action_dim=action_dim,
        hidden_sizes=[M, M],
    )
    eval_policy = MakeDeterministic(policy)
    agent_trainer = SACTrainer(
        env=eval_env,
        policy=policy,
        qf1=qf1,
        qf2=qf2,
        target_qf1=target_qf1,
        target_qf2=target_qf2,
        **variant['agent_trainer_kwargs']
    )
    # Path collectors.
    eval_path_collector = MdpPathCollector(
        eval_env,
        eval_policy,
    )
    expl_path_collector = MdpPathCollector(
        expl_env,
        policy,
    )
    model_path_collector = MdpPathCollector(
        model_env,
        policy,
    )
    # Create algorithm.
    algo = TorchDynaRLAlgorithm(
        agent_trainer,
        model_trainer,
        expl_env,
        eval_env,
        model_env,
        expl_path_collector,
        eval_path_collector,
        model_path_collector,
        replay_buffer,
        imaginary_replay_buffer,
        **variant['algorithm_kwargs'],
    )
    algo.to(ptu.device)
    algo.train()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_id', default='test')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--cuda_device', default='')
    parser.add_argument('--pudb', action='store_true')
    args = parser.parse_args()
    variant = deepcopy(DEFAULT_VARIANT)
    variant['run_id'] = args.run_id
    variant['seed'] = args.seed
    variant['cuda_device'] = args.cuda_device
    if args.pudb:
        import pudb; pudb.set_trace()
    launch_variant(variant)
