#!/usr/bin/env python

import datetime
import json
import matplotlib.pyplot as plt
import os
import pandas as pd


def load_json(filename):
    """
    Loads a json file
    Returns a Python dictionary
    """
    with open(filename) as f:
        return json.load(f)


def print_dict(d, indent=1, dest="stdout"):
    """
    Pretty prints a Python dictionary
    to stdout or a filename
    """
    json_str = json.dumps(d, indent=indent)
    if dest == "stdout":
        print(json_str)
    elif isinstance(dest, str):
        with open(dest, 'w') as f:
            f.write(json_str)


def dict_to_tuplelist(d):
    """
    Converts a dictionary of dictionaries
    into a list of tuples
    """
    tuple_list = []
    data = []
    for k, v in d.items():
        for w, x in v.items():
            tuple_list.append((k, w))
            data.append(x)
    return tuple_list, data


def dict_to_dataframe(d):
    """
    Converts a dictionary of dictionaries
    into a Pandas dataframe, with a MultiIndex column header
    """
    tuple_list, data = dict_to_tuplelist(d)
    idx = pd.MultiIndex.from_tuples(tuple_list)
    return pd.DataFrame([data], columns=idx)


if __name__ == "__main__":
    pass

