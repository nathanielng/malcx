#!/usr/bin/env python

# Plots hyperopt distributions

import argparse
import hyperopt
import matplotlib.pyplot as plt
import numpy as np

from hyperopt.pyll.stochastic import sample


def sample_space(space, n):
    """
    Samples the space provided
    """
    samples = []
    for _ in range(n):
        samples.append(sample(space))
    return samples


def plot_sample_space_1D(samples, title=None, filename=None):
    """
    Plots the samples
    """
    fig, ax = plt.subplots(1, 1, figsize=(7,5))
    ax.hist(samples, bins=20, edgecolor='black')
    ax.set_xlabel('x')
    ax.set_ylabel('Frequency')
    if title is not None:
        ax.set_title(title)
    if isinstance(filename, str):
        plt.savefig(filename)


def get_space(space_type, param1, param2):
    if space_type == 'normal':
        space = hyperopt.hp.normal('x', float(param1), float(param2))
    elif space_type == 'uniform':
        space = hyperopt.hp.uniform('x', float(param1), float(param2))
    elif space_type == 'quniform':
        space = hyperopt.hp.quniform('x', float(param1), float(param2))
    elif space_type == 'arange':
        space = hyperopt.hp.choice('x', np.arange(int(param1), int(param2), dtype=int))
    else:
        space = None
    return space


def plot_space(space_type, param1, param2, n=1000):
    space = get_space(space_type, param1, param2)
    if space is not None:
        samples = sample_space(space, n)
        title = f'{space_type}({param1}, {param2})'
        plot_sample_space_1D(samples, title, 'hyperopt.png')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('space')
    parser.add_argument('p1')
    parser.add_argument('p2')
    parser.add_argument('--n', default=1000)
    args = parser.parse_args()
    plot_space(args.space, args.p1, args.p2, args.n)

# Examples:
# python hyperopt_dist.py 'normal' 4.0 1.0
# python hyperopt_dist.py 'arange' 1 3
# python hyperopt_dist.py 'uniform' 1 2
