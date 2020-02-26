from rlkit.samplers.rollout_functions import rollout
from rlkit.torch.pytorch_util import set_gpu_mode
import argparse
import torch
import uuid
from rlkit.core import logger
import sys
import os
from ipdb import set_trace as db
from tqdm import trange
import pickle as p
sys.path.append('../TokamakModels4RL')
from gym_envs.generic.SISO_Feedback_SimpleConfinement import Scenario



def simulate_policy(args):
    data = torch.load(args.file)
    policy = data['evaluation/policy']
    if 'evaluation/env' in data:
        env = data['evaluation/env']
    else:
        env = Scenario()

    print("Policy loaded")
    if args.gpu:
        set_gpu_mode(True)
        policy.cuda()
    for _ in trange(2000):
        filename = str(uuid.uuid4()) + '.pkl'
        path = rollout(
            env,
            policy,
            max_path_length=args.H,
            render=False,
        )
        if hasattr(env, "log_diagnostics"):
            env.log_diagnostics([path])
        logger.dump_tabular()
        with open(filename, 'wb') as f:
            p.dump(path, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('file', type=str,
                        help='path to the snapshot file')
    parser.add_argument('--H', type=int, default=1000,
                        help='Max length of rollout')
    parser.add_argument('--gpu', action='store_true')
    args = parser.parse_args()

    simulate_policy(args)
