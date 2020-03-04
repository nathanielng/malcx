#!/usr/bin/env python

# Description:
# Postprocessing subroutines

import argparse
import glob
import matplotlib.pyplot as plt
import os
import pandas as pd
import re


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
    best_coeff = df[col].min()
    idxmin = df[col].idxmin()

    print(f'(n={len(df)})')
    print(f"   - Best coefficient of determination = {best_coeff} (at idx={idxmin})")
    print(f"   - Best set of parameters:")
    print_params(df.loc[idxmin, :].T)
    return df


def plot_parameter_histograms(df, filename=None):
    df.hist(figsize=(12,12))
    if filename is not None:
        plt.savefig(filename)


def plot_parameter_history(df, filename=None):
    if len(df.columns) != 12:
        columns = ','.join(df.columns)
        print('Skipping because columns != 12: {columns}')
        return
    df.plot.line(marker='o', linestyle='', figsize=(12,12), layout=(4, 3), subplots=True)
    if filename is not None:
        plt.savefig(filename)


def load_history_files(folder):
    print(f'Run Folder: {folder}')
    files = glob.glob(f'{folder}/parameter_history*.csv')
    for i, file in enumerate(files):
        basename = os.path.basename(file)
        img_file, _ = os.path.splitext(basename)
        img_file1 = f'images/{img_file}_histogram.png'
        img_file2 = f'images/{img_file}_history.png'
        print(f'{i}: {basename} ', end='')
        df = summarize_historyfile(file)

        if os.path.isfile(img_file1):
            continue
        plot_parameter_histograms(df, img_file1)
        plot_parameter_history(df, img_file2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--history', default=None, help='History folder for parameters')
    args = parser.parse_args()

    if args.history is not None:
        load_history_files(args.history)
    else:
        load_history_files(RUN_FOLDER)
