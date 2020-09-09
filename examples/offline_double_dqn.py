"""
Run offline double dqn on cartpole.
"""
import numpy as np
import gym
from torch import nn as nn

from rlkit.exploration_strategies.base import \
    PolicyWrappedWithExplorationStrategy
from rlkit.exploration_strategies.epsilon_greedy import EpsilonGreedy
from rlkit.policies.argmax import ArgmaxDiscretePolicy
from rlkit.torch.dqn.dqn import DQNTrainer
from rlkit.torch.networks import Mlp
import rlkit.torch.pytorch_util as ptu
from rlkit.data_management.offline_data_store import OfflineDataStore
from rlkit.launchers.launcher_util import setup_logger
from rlkit.samplers.data_collector import MdpPathCollector
from rlkit.torch.torch_rl_algorithm import TorchOfflineRLAlgorithm


def experiment(variant):
    """Run the experiment."""
    eval_env = gym.make('CartPole-v0')
    obs_dim = eval_env.observation_space.low.size
    action_dim = eval_env.action_space.n
    # Collect data.
    print('Collecting data...')
    data = []
    while len(data) < variant['offline_data_size']:
        done = False
        s = eval_env.reset()
        while not done:
            a = np.random.randint(action_dim)
            n, r, done, _ = eval_env.step(a)
            one_hot_a = np.zeros(action_dim)
            one_hot_a[a] = 1
            data.append((s, one_hot_a, r, n, done))
            s = n
            if len(data) == variant['offline_data_size']:
                break

    qf = Mlp(
        hidden_sizes=[32, 32],
        input_size=obs_dim,
        output_size=action_dim,
    )
    target_qf = Mlp(
        hidden_sizes=[32, 32],
        input_size=obs_dim,
        output_size=action_dim,
    )
    qf_criterion = nn.MSELoss()
    eval_policy = ArgmaxDiscretePolicy(qf)
    eval_path_collector = MdpPathCollector(
        eval_env,
        eval_policy,
    )
    trainer = DQNTrainer(
        qf=qf,
        target_qf=target_qf,
        qf_criterion=qf_criterion,
        **variant['trainer_kwargs']
    )
    offline_data = OfflineDataStore(data=data,)
    algorithm = TorchOfflineRLAlgorithm(
        trainer=trainer,
        evaluation_env=eval_env,
        evaluation_data_collector=eval_path_collector,
        offline_data=offline_data,
        **variant['algorithm_kwargs']
    )
    algorithm.to(ptu.device)
    algorithm.train()




if __name__ == "__main__":
    # noinspection PyTypeChecker
    variant = dict(
        algorithm="SAC",
        version="normal",
        layer_size=256,
        offline_data_size=int(1e6),
        algorithm_kwargs=dict(
            num_epochs=3000,
            batch_size=2048,
            num_eval_steps_per_epoch=5000,
            num_train_loops_per_epoch=1000,
            max_path_length=200,
        ),
        trainer_kwargs=dict(
            discount=0.99,
            learning_rate=3E-4,
        ),
    )
    setup_logger('offline-cartpole', variant=variant)
    ptu.set_gpu_mode(True)  # optionally set the GPU (default=False)
    experiment(variant)
