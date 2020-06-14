from collections import deque
import numpy as np
import torch

from rlkit.policies.base import Policy
from rlkit.torch.PETS.optimizer import CEMOptimizer, RSOptimizer
from rlkit.torch.core import np_ify
from ipdb import set_trace as db


class MPCPolicy(Policy):
    """
    Usage:
    ```
    policy = MPCPolicy(...)
    action, mean, log_std, _ = policy(obs)
    ```
    """
    def __init__(
            self,
            model,
            obs_dim,
            action_dim,
            num_particles,
            cem_horizon,
            cem_iters,
            cem_popsize,
            cem_num_elites,
            sampling_strategy,
            optimizer='CEM',
            opt_freq=1,
            cem_alpha=0.1,
    ):
        super().__init__()
        assert sampling_strategy in ('TS1', 'TSinf'), "Sampling Strategy must be TS1 or TSinf"
        assert optimizer in ('CEM', 'RS'), "Only CEM and RS optimizers supported"
        self.model = model
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        # set up CEM optimizer
        sol_dim = action_dim * cem_horizon
        # assuming normalized environment
        self.ac_ub = 1
        self.ac_lb = -1
        if optimizer == 'CEM':
            self.optimizer = CEMOptimizer(
                sol_dim,
                cem_iters,
                cem_popsize,
                cem_num_elites,
                self._cost_function,
                upper_bound=self.ac_ub,
                lower_bound=self.ac_lb,
                alpha=cem_alpha)
        elif optimizer == 'RS':
            popsize = cem_popsize * cem_iters
            self.optimizer = RSOptimizer(
                sol_dim,
                self._cost_function,
                popsize,
                upper_bound=self.ac_ub,
                lower_bound=self.ac_lb)

        self.cem_horizon = cem_horizon
        # 16 here comes from torch PETS implementation, unsure why
        self.cem_init_var = np.tile(np.square(self.ac_ub - self.ac_lb) / 16, [self.cem_horizon * self.action_dim])
        self.current_obs = None
        self.prev_sol = np.tile(
                (self.ac_lb + self.ac_ub)/ 2,
                [self.cem_horizon],
        )
        self.num_particles = num_particles
        self.sampling_strategy = sampling_strategy
        self.action_buff = deque()
        self.opt_freq = opt_freq

    def reset(self):
        self.optimizer.reset()
        self.current_obs = None
        self.prev_sol = np.tile(
                (self.ac_lb + self.ac_ub)/ 2,
                [self.cem_horizon],
        )
        self.action_buff = deque()

    def sample_action(self):
        return np.random.uniform(low=self.ac_lb, high=self.ac_ub, size=self.action_dim)

    def get_action(self, obs_np):
        if not self.model.trained_at_all:
            return self.sample_action(), {}
        if len(self.action_buff) > 0:
            return np.asarray([self.action_buff.popleft()]), {}
        self.current_obs = obs_np
        new_sol = self.optimizer.obtain_solution(
                self.prev_sol,
                self.cem_init_var,
        )
        self.prev_sol = np.concatenate([
                np.copy(new_sol)[self.action_dim * self.opt_freq:],
                np.zeros(self.opt_freq),
        ])
        for a in new_sol[1:self.opt_freq]:
            self.action_buff.append(a)
        return np.asarray([new_sol[0]]), {}

    def get_actions(self, obs_np, deterministic=False):
        # TODO: figure out how this is used
        # return eval_np(self, obs_np, deterministic=deterministic)[0]
        raise NotImplementedError()

    @torch.no_grad()
    def _cost_function(self, ac_seqs):
        '''
        a function from action sequence to cost, either from the model or the given
        cost function. TODO: add the sampling strategies from the PETS paper

        ac_seqs: batch_size * (cem_horizon * action_dimension)
        requires self.current_obs to be accurately set
        '''
        batch_size = ac_seqs.shape[0]
        ac_seqs = ac_seqs.reshape((batch_size, self.cem_horizon, self.action_dim))
        obs = np.tile(self.current_obs, reps=(batch_size * self.num_particles, 1))
        ac_seqs = np.tile(ac_seqs[:, np.newaxis, :, :], reps=(1, self.num_particles, 1, 1))
        ac_seqs = ac_seqs.reshape((batch_size * self.num_particles, self.cem_horizon, self.action_dim))
        observations, rewards = self.model.unroll(obs, ac_seqs, self.sampling_strategy)
        rewards = np_ify(rewards).reshape((batch_size, self.num_particles, self.cem_horizon))
        # sum over time, average over particles
        # TODO (maybe): add discounting
        return -rewards.sum(axis=(2)).mean(axis=(1))
