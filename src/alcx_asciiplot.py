#!/usr/bin/env python

# Requirements:
# pip install drawilleplot

import argparse
import glob
import numpy as np
import os

import matplotlib
matplotlib.use('module://drawilleplot')

from alcx_plot import add_timestamps, summarize_historyfile
from alcx_reader import find_file_by_index, RUN_FOLDER
from matplotlib import pyplot as plt


def draw_plot(ax, x, y, label, xlabel=None, ylabel=None, title=None):
    ax.scatter(x, y, s=15, c="r", alpha=0.5, marker='x', label=label)
    if title is not None:
        ax.set_title(title)
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    ax.axis([x.min(), x.max(), y.min(), y.max()])
    ax.legend(loc='best')


def process_history_file(filename):
    if filename is None:
        filename = find_file_by_index(-1)
        print(f'Using file: {filename}')

    df = summarize_historyfile(filename)

    y = df['coefficient of determination']
    cols = ['distribution samples', 'gradient directions', 'data for gradient',
       'data for intercept', 'iteration layers', 'extra data filters',
       'input nodes', 'fraction divisions tried',
       'minimum input node correlation', 'maximum input node intercorrelation',
       'uncertainty']
    
    for col in cols:
        x = df[col]

        fig, ax = plt.subplots(1, 1, figsize=(5, 4))
        draw_plot(ax, x, y, xlabel=col,
            label='coefficient of determination')
        plt.show()
        plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', default=None, help='Parameter history file')
    args = parser.parse_args()
    process_history_file(args.file)
