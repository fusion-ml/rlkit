"""
Train an ensemble of PNNs for modelling.
"""
import argparse
import os

import h5py
import numpy as np
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader, TensorDataset

from dde.estimators import PNN
from dde.model_trainer import ModelTrainer

def load_data(args):
    hdata = h5py.File(args.data_path, 'r')
    obs = hdata['observations'][()]
    acts = hdata['actions'][()][:-1]
    rews = hdata['rewards'][()][:-1].reshape(-1, 1)
    st = obs[:-1]
    st1 = obs[1:]
    x_dim = st.shape[1] + acts.shape[1]
    y_dim = st.shape[1] + 1
    x_means = torch.cat([
        torch.Tensor(np.mean(st, axis=0)),
        torch.zeros(acts.shape[1]),
    ])
    x_stds = torch.cat([
        torch.Tensor(np.std(st, axis=0)),
        torch.ones(acts.shape[1]),
    ])
    x_data = torch.Tensor(np.hstack([st, acts]))
    y_data = torch.cat([
        torch.Tensor(rews),
        torch.Tensor(st1 - st),
    ], dim=1)
    standardizers = [(x_means, x_stds),
            (torch.mean(y_data, dim=0), torch.std(y_data, dim=0))]
    return x_dim, y_dim, x_data, y_data, standardizers

def s2i(string):
    if ',' not in string:
        return []
    return [int(s) for s in string.split(',')]

def train_ensemble(args):
    x_dim, y_dim, x_data, y_data, standardizers = load_data(args)
    use_cpu = args.cuda_device == ''
    for ens_idx in range(1, args.n_members + 1):
        print('=========ENSEMBLE MEMBER %d========' % ens_idx)
        tr_x, val_x, tr_y, val_y = train_test_split(x_data, y_data,
                test_size=args.val_prop, random_state=args.seed + ens_idx)
        tr_dataset = DataLoader(
            TensorDataset(tr_x, tr_y),
            batch_size=args.bs,
            shuffle=use_cpu,
            pin_memory=not use_cpu,
        )
        val_dataset = DataLoader(
            TensorDataset(tr_x, tr_y),
            batch_size=args.bs,
            shuffle=use_cpu,
            pin_memory=not use_cpu,
        )
        pnn = PNN(
            input_dim=x_dim,
            encoder_hidden_sizes=s2i(args.encoder_hidden),
            latent_dim=args.latent_dim,
            mean_hidden_sizes=s2i(args.mean_hidden),
            logvar_hidden_sizes=s2i(args.logvar_hidden),
            output_dim=y_dim,
            logvar_bounds=[args.logvar_lower, args.logvar_upper],
            bound_loss_coef=args.bound_loss_coef,
        )
        pnn.set_standardization(standardizers)
        trainer = ModelTrainer(
                model=pnn,
                learning_rate=args.lr,
                cuda_device=args.cuda_device,
                save_path=os.path.join(args.save_path, 'member_%d' % ens_idx),
                save_freq=100,
        )
        trainer.fit(tr_dataset, args.epochs, val_dataset, args.od_wait)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_path')
    parser.add_argument('--data_path')
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--od_wait', type=int, default=50)
    parser.add_argument('--encoder_hidden', default='400,400,400,400')
    parser.add_argument('--mean_hidden', default='')
    parser.add_argument('--logvar_hidden', default='')
    parser.add_argument('--latent_dim', type=int, default=400)
    parser.add_argument('--logvar_lower', type=float, default=-10)
    parser.add_argument('--logvar_upper', type=float, default=0.5)
    parser.add_argument('--bound_loss_coef', type=float, default=1e-2)
    parser.add_argument('--n_members', type=int, default=5)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--wd', type=float, default=0)
    parser.add_argument('--bs', type=int, default=256)
    parser.add_argument('--val_prop', type=float, default=0.1)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--cuda_device', type=str, default='')
    parser.add_argument('--pudb', action='store_true')
    args = parser.parse_args()
    if args.pudb:
        import pudb; pudb.set_trace()
    return args

if __name__ == '__main__':
    train_ensemble(parse_args())
