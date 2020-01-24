#!/usr/bin/env python

import argparse
import datetime
import hyperopt
import json
import matplotlib.pyplot as plt
import os
import pandas as pd
import re
import subprocess


def load_json(filename):
    """
    Loads a json file
    Returns a Python dictionary
    """
    with open(filename) as f:
        return json.load(f)


def print_json(d, indent=1, dest="stdout"):
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


def run_job(path):
    process = subprocess.Popen(
        [path],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()
    return stdout.decode(), stderr.decode()


def get_coefficient_of_determination(txt):
    m = re.search(r'Coefficient of determination\s*([0-9.]+)\s*±\s*([0-9.]+)', txt)
    if m is None:
        return None
    else:
        return {
            'coeff': float(m.group(1)),
            'uncertainty': float(m.group(2))
        }


def evalution_function(params, json_file, exec_path):
    write_dict(params, json_file)
    stdout, stderr = run_job(exec_path)
    result = get_coefficient_of_determination(stdout)
    if result is not None:
        return result['coeff']
    else:
        return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--exec_path')
    parser.add_argument('--json_file')
    args = parser.parse_args()

