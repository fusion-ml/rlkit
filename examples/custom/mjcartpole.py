"""Copied from PETS"""
import os
import torch

import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env
import rlkit.torch.pytorch_util as ptu


class CartpoleEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    PENDULUM_LENGTH = 0.6

    def __init__(self):
        utils.EzPickle.__init__(self)
        xml_path = os.path.join(
                os.getenv("RLKIT_PATH"),
                'examples/custom/assets/cartpole.xml',
        )
        mujoco_env.MujocoEnv.__init__(self, xml_path, 2)

    def step(self, a):
        self.do_simulation(a, self.frame_skip)
        ob = self._get_obs()

        cost_lscale = CartpoleEnv.PENDULUM_LENGTH
        reward = np.exp(
                -np.sum(np.square(self._get_ee_pos(ob) - np.array([0.0, CartpoleEnv.PENDULUM_LENGTH]))) / (cost_lscale ** 2)
        )
        reward -= 0.01 * np.sum(np.square(a))

        done = False
        return ob, reward, done, {}

    def reset_model(self):
        qpos = self.init_qpos + np.random.normal(0, 0.1, np.shape(self.init_qpos))
        qvel = self.init_qvel + np.random.normal(0, 0.1, np.shape(self.init_qvel))
        self.set_state(qpos, qvel)
        return self._get_obs()

    def _get_obs(self):
        return np.concatenate([self.sim.data.qpos.flat, self.sim.data.qvel.flat]).ravel()

    @staticmethod
    def _get_ee_pos(x):
        x0, theta = x[0], x[1]
        return np.array([
            x0 - CartpoleEnv.PENDULUM_LENGTH * np.sin(theta),
            -CartpoleEnv.PENDULUM_LENGTH * np.cos(theta)
        ])

    def viewer_setup(self):
        v = self.viewer
        v.cam.trackbodyid = 0
        v.cam.distance = v.model.stat.extent

def np_get_cp_reward(curr_obs, action, next_obs):
    x0s, thetas = next_obs[:, 0], next_obs[:, 1]
    ee_poss = np.hstack([
        x0s - CartpoleEnv.PENDULUM_LENGTH * np.sin(thetas),
        -CartpoleEnv.PENDULUM_LENGTH * np.cos(thetas)])
    cost_lscale = CartpoleEnv.PENDULUM_LENGTH
    tgt = np.asarray([0.0, CartpoleEnv.PENDULUM_LENGTH]).reshape(1, -1)
    sqrd = np.square(ee_poss - tgt)
    return (np.exp(-np.sum(sqrd, axis=1) / (cost_lscale ** 2))
            - np.sum(action ** 2, axis=1) * 0.01)

@torch.no_grad()
def torch_get_cp_reward(curr_obs, action, next_obs):
    x0s, thetas = next_obs[:, 0], next_obs[:, 1]
    ee_poss = torch.stack([
        x0s - CartpoleEnv.PENDULUM_LENGTH * torch.sin(thetas),
        -CartpoleEnv.PENDULUM_LENGTH * torch.cos(thetas)], dim=1).to(ptu.device)
    cost_lscale = CartpoleEnv.PENDULUM_LENGTH
    tgt = torch.Tensor([0.0, CartpoleEnv.PENDULUM_LENGTH]).to(ptu.device)
    sqrd = (ee_poss - tgt) ** 2
    return torch.exp(-torch.sum(sqrd, 1) / (cost_lscale ** 2)) - torch.sum(action ** 2, 1) * 0.01
