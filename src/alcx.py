#!/usr/bin/env python

import argparse
import datetime
import hyperopt
import json
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import re
import subprocess


# ----- Set Parameters -----
ALCX_RUN_FOLDER = os.getenv('ALCX_RUN_FOLDER', None)
ALCX_EXECUTABLE = os.getenv('ALCX_EXECUTABLE', None)
ALCX_JSONFILE = None
ALCX_OUTPUTLOG = None
PARAMS0 = {}


# ----- Subroutines -----
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
    """
    Returns the search space in a dictionary of hyperopt distributions

    Parameters that do not require a search space:
     - 'data in gradient': None
     - 'data in intercept': None
     - 'descriptor columns': (fixed)
     - 'model compression': None

    """
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
        print("----- Could not find coefficient of determination -----")
        print("**Output**:")
        print(txt)
        print("-------------------------------------------------------")
        return None
    else:
        return {
            'coeff': float(m.group(1)),
            'uncertainty': float(m.group(2))
        }


def evaluation_function(params):
    # Overwrite PARAMS0 with params
    params2 = PARAMS0
    params2['network'] = {
        **PARAMS0['network'],
        'distribution samples': int(params['distribution samples']),
        'gradient directions': int(params['gradient directions']),
        'data for gradient': int(params['data for gradient']),
        'data for intercept': int(params['data for intercept']),
        'iteration layers': int(params['iteration layers']),
        'extra data filters': int(params['extra data filters'])
    }
    params2['training'] = {
        **PARAMS0['training'],
        'input nodes': int(params['input nodes']),
        'fraction divisions tried': params['fraction divisions tried'],
        'minimum input node correlation': params['minimum input node correlation'],
        'maximum input node correlation': params['maximum input node correlation']
    }
    evaluation_function.ITERATION += 1
    print(f'---------- Iteration: {evaluation_function.ITERATION} ----------')
    print_json(params2, dest=ALCX_JSONFILE)
    print_json(params2, dest='stdout')
    stdout, stderr = run_job(ALCX_EXECUTABLE)
    if ALCX_OUTPUTLOG is not None:
        with open(ALCX_OUTPUTLOG, 'w') as f:
            f.write(stdout)
    result = get_coefficient_of_determination(stdout)
    if result is not None:
        print(f"Result: {result}")
        return -result['coeff']
    else:
        print(f"Result: {result}")
        return 0.0


def trials2df(result, trials):
    x = pd.Series(trials.idxs_vals[1]['x'])
    loss = pd.Series([x['loss'] for x in trials.results])
    df = pd.DataFrame({
        'x': x,
        'loss': loss
        }, index = trials.idxs_vals[0]['x'])
    return df


def hyperopt_optimize(opt_function, space,
                      algo=hyperopt.tpe.suggest):
    """
    Finds minimum of `opt_function` in `space`.
    Algorithm defaults to TPE (Tree of Parzen Estimators)
    """
    trials = hyperopt.Trials()
    result = hyperopt.fmin(
        fn=opt_function,
        space=space,
        algo=algo,
        trials=trials,
        max_evals=100,
        rstate=np.random.RandomState(100)
    )
    return {
        'result': result,
        'trials': trials
    }


evaluation_function.ITERATION = 0

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--exec_path', default=ALCX_EXECUTABLE)
    parser.add_argument('--run_folder', default=ALCX_RUN_FOLDER)
    args = parser.parse_args()
    ALCX_EXECUTABLE = args.exec_path
    ALCX_RUN_FOLDER = args.run_folder

    if (ALCX_RUN_FOLDER is None) or (ALCX_EXECUTABLE is None):
        print('The environment variables ALCX_RUN_FOLDER and ALCX_EXECUTABLE')
        print('need to be defined.')

    ALCX_OUTPUTLOG = os.path.join(ALCX_RUN_FOLDER, 'alcx.out')
    ALCX_JSONFILE = os.path.join(ALCX_RUN_FOLDER, 'input.json')
    ALCX_JSONFILE_ORIGINAL = os.path.join(ALCX_RUN_FOLDER, 'input_original.json')

    # ----- Load JSON File -----
    PARAMS0 = load_json(ALCX_JSONFILE_ORIGINAL)
    print(f'---------- Starting JSON File ----------')
    print_json(PARAMS0)

    # ----- Define Parameter Space -----
    input_nodes = PARAMS0['training']['input nodes']
    descriptor_columns = PARAMS0['training']['descriptor columns']
    space = get_search_space(input_nodes=input_nodes, n_columns=descriptor_columns)

    # ----- Optimize -----
    try:
        os.chdir(ALCX_RUN_FOLDER)
        # result = evaluation_function(params)
        # print(f'Result: {result}')

        result = hyperopt_optimize(evaluation_function, space)
        print(f"Result = {result['result']}")

    except Exception as e:
        print(f"Unable to change folder to {ALCX_RUN_FOLDER}")
        print(f"Exception: {e}")
        quit()

    # ----- Save results -----
    df = trials2df(result['result'], result['trials'])
    df.to_csv('results.csv')
    idxmin = df['loss'].idxmin()
    print(f"Best x: {df.loc[idxmin, 'x']}")
    print(f"with loss = {df['loss'].min()} (at iteration # {idxmin})")
