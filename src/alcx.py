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


JSON_FILE = os.getenv('JSON_FILE', None)
EXEC_PATH = os.getenv('EXEC_PATH', None)
OUTPUTLOG = os.getenv('OUTPUTLOG', None)


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


def get_search_space(input_nodes, n_columns):
    d = {
        'distribution samples': hyperopt.hp.uniform('distribution', 10, 1000),
        'gradient directions': hyperopt.hp.uniform('gradient directions', 0, input_nodes - 1),
        'data for gradient': hyperopt.hp.uniform('data for gradient', 1, 10*input_nodes),
        'data for intercept': hyperopt.hp.uniform('data for intercept', 1, 10),
        'iteration layers': hyperopt.hp.uniform('iteration layers', 1, 2),
        'extra data filters': hyperopt.hp.uniform('extra data filters', -1, 1),
        'input nodes': hyperopt.hp.uniform('input nodes', 1, n_columns - 1 ),
        'fraction divisions tried': hyperopt.hp.uniform('fraction divisions tried', 0, 1),
        'minimum input node correlation': hyperopt.hp.uniform('minimum input node correlation', 0, 1),
        'maximum input node correlation': hyperopt.hp.uniform('maximum input node correlation', 0, 1),
    }
    # 'data in gradient': None,
    # 'data in intercept': None,
    # 'descriptor columns': 'fixed',
    # 'model compression': None
    return d


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


def evaluation_function(params):
    print_json(params, dest=JSON_FILE)
    stdout, stderr = run_job(EXEC_PATH)
    result = get_coefficient_of_determination(stdout)
    if result is not None:
        return result['coeff']
    else:
        return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--exec_path', default=None)
    parser.add_argument('--json_file', default=None)
    args = parser.parse_args()
    JSON_FILE = args.json_file
    EXEC_PATH = args.exec_path

    if (JSON_FILE is None) or (EXEC_PATH is None):
        print('The environment variables JSON_FILE and/or EXEC_PATH')
        print('need to be defined.')

    params = load_json(JSON_FILE)
    print_json(params)
    result = evaluation_function(params)
    print(f'Result = {result}')
