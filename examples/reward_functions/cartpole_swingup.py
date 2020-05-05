import torch


@torch.no_grad()
def cartpole_swingup_reward_v1(state, action, next_state):
    return (1 + next_state[..., 2]) / 2
