#!/usr/bin/env python

import pandas as pd


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


if __name__ == "__main__":
    pass
