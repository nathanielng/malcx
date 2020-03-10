#!/usr/bin/env python

# Description:
# Postprocessing subroutines

import argparse
import datetime
import glob
import matplotlib.pyplot as plt
import os
import pandas as pd
import re

from matplotlib.cm import get_cmap
from pandas.plotting import scatter_matrix

RUN_FOLDER = os.path.abspath('../../run/')


def read_parameter_history_file(filename):
    df = pd.read_csv(filename, index_col=0, header=[0, 1])
    df = df.reset_index(drop=True)
    df.index = df.index.set_names('Iteration')
    col = ('results', 'coefficient of determination')
    if col in df.columns:
        df = df.sort_values(col, ascending=False)
    return df


def remove_first_row_header(df):
    """
    In a dataframe with a multi-index header,
    Use only the second row of headers (first row is removed)
    """
    header2 = df.T.reset_index().iloc[:, 1].values
    df.columns = header2
    return df


def print_params(s):
    for param, value in s.iteritems():
        print(f'       {param}: {value}')


def add_timestamps(filenames, fmt=r'parameter_history_.*([0-9]{8}-[0-9]{4})h.csv'):
    file_list = []
    for filename in filenames:
        m = re.search(fmt, filename)
        if m is not None:
            ts = datetime.datetime.strptime(m.group(1), '%Y%m%d-%H%M')
            file_list.append([filename, ts])
    return pd.DataFrame(file_list, columns=['Filename', 'Date']).sort_values('Date')


def summarize_historyfile(filename):
    """
    Summarizes a parameter history file
    """
    df = read_parameter_history_file(filename)
    df = remove_first_row_header(df)

    drop_cols = ['task list', 'data in gradient', 'data in intercept', 'descriptor columns', 'model compression']
    try:
        df = df.drop(labels=drop_cols, axis=1)
    except Exception as e:
        print(f'Exception: {e}')
        print(f'Columns: {df.columns}')

    col = 'coefficient of determination'
    best_coeff = df[col].max()
    idxmax = df[col].idxmax()

    print(f'(n={len(df)})')
    print(f"   - Best coefficient of determination = {best_coeff} (at idx={idxmax})")
    print(f"   - Best set of parameters:")
    print_params(df.loc[idxmax, :].T)
    return df


def plot_parameter_histograms(df, filename=None):
    df.hist(figsize=(12,12))
    if filename is not None:
        plt.savefig(filename)


def plot_parameter_history(df, filename=None):
    df.plot.line(marker='o', linestyle='', figsize=(12,12), layout=(4, 3), subplots=True)
    if filename is not None:
        plt.savefig(filename)


def plot_parameter_correlation(df, filename=None):
    n_rows, n_cols = 4, 3
    colors = get_cmap('Paired').colors
    cod_idx = df.columns.get_loc('coefficient of determination')
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(15, 12))
    i = 0
    for row in range(n_rows):
        for col in range(n_cols):
            axs[row, col].plot(
                df.iloc[:, i],
                df.iloc[:, cod_idx],
                'o',
                color=colors[i],
                label=df.columns[i])
            axs[row, col].legend(frameon=False)
            i += 1
    if filename is not None:
        plt.savefig(filename)


def plot_parameter_correlation2(df, filename=None):
    df_new = df.set_index('coefficient of determination')
    df_new.plot.line(marker='o', linestyle='', figsize=(12,12), layout=(4, 3), subplots=True)
    if filename is not None:
        plt.savefig(filename)


def plot_scatter_matrix(df, filename=None):
    scatter_matrix(df, figsize=(20,20));
    if filename is not None:
        plt.savefig(filename)


def load_history_files(folder, refresh):
    print(f'Run Folder: {folder}')
    files = glob.glob(f'{folder}/parameter_history*.csv')
    for i, file in enumerate(files):
        basename = os.path.basename(file)
        img_file, _ = os.path.splitext(basename)
        img_file1 = f'images/{img_file}_histogram.png'
        img_file2 = f'images/{img_file}_history.png'
        print(f'{i}: {basename} ', end='')
        df = summarize_historyfile(file)

        if refresh is False and os.path.isfile(img_file1):
            continue
        plot_parameter_histograms(df, img_file1)
        plot_parameter_history(df, img_file2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--history', default=None, help='History folder for parameters')
    parser.add_argument('--refresh', action='store_true', help='Force image refresh')
    args = parser.parse_args()

    if args.history is not None:
        load_history_files(args.history, args.refresh)
    else:
        load_history_files(RUN_FOLDER, args.refresh)