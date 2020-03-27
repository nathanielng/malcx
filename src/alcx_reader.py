#!/usr/bin/env python

import argparse
import datetime
import glob
import json
import os
import numpy as np
import pandas as pd
import re

from alcx import print_json


def get_run_folder():
    folder = os.getenv('ALCX_RUN_FOLDER', None)
    if folder is not None:
        return os.path.abspath(folder)

    for folder in ['../../run/', '../run/']:
        if os.path.isdir(folder):
            return os.path.abspath(folder)

    print('Create a run folder in ../../run or ../run')
    print('Or set the environment variable: RUN_FOLDER')
    return None


def add_timestamps(filenames, fmt=r'parameter_history_.*([0-9]{8}-[0-9]{4})h.csv'):
    file_list = []
    for filename in filenames:
        m = re.search(fmt, filename)
        if m is not None:
            ts = datetime.datetime.strptime(m.group(1), '%Y%m%d-%H%M')
            file_list.append([filename, ts])
    return pd.DataFrame(file_list, columns=['Filename', 'Date']).sort_values('Date')


def find_file_by_index(idx):
    """
    Find file (of the form `parameter_history*.csv`) by index
    where 0 = first file,
          1 = second file,
         -2 = second last file
         -1 = last file
    """
    files = glob.glob(f'{RUN_FOLDER}/parameter_history*.csv')
    file_list = add_timestamps(files)
    if len(file_list) == 0:
        print('No files in folder')
        return None
    else:
        return file_list.iloc[idx].loc['Filename']


def df_row_to_dict(df_slice):
    x = df_slice.reset_index().set_index('level_0')
    keys = x.index.unique()

    d = {}
    for key in keys:
        if key == 'results':
            continue

        y = x[x.index == key].set_index('level_1')
        z = y.iloc[:, 0].to_dict()
        for k, v in z.items():
            if isinstance(v, np.int64):
                z[k] = int(v)
        d[key] = z
    return d


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


RUN_FOLDER = get_run_folder()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--index', default=-1, type=int)
    parser.add_argument('--json', default=None)
    args = parser.parse_args()
    filename = find_file_by_index(args.index)
    print(f'Parameter file: {filename}')

    if filename is None:
        quit()

    df = read_parameter_history_file(filename)
    df_best = df.iloc[0, :]
    dict_best = df_row_to_dict(df_best)
    if isinstance(args.json, str):
        print_json(dict_best, dest=args.json)
    else:
        print_json(dict_best, dest='stdout')
