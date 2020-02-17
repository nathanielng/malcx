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

DATETIME_STAMP = datetime.datetime.now().strftime('%Y%m%d-%H%Mh')
RESULT_FILE = os.path.join(ALCX_RUN_FOLDER, f'result-{DATETIME_STAMP}.pkl')
TRIAL_FILE = os.path.join(ALCX_RUN_FOLDER, f'trials-{DATETIME_STAMP}.pkl')
RESULTS_CSV_FILE = os.path.join(ALCX_RUN_FOLDER, f'result-{DATETIME_STAMP}.csv')
PARAMETER_HISTORY_FILE = 'parameter_history_{DATETIME_STAMP}.csv'
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
        'maximum input node intercorrelation': hyperopt.hp.uniform('maximum input node correlation', 0, 1),
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
    m = re.search(r'Coefficient of determination\s*(-?[0-9.]+)\s*±\s*([0-9.]+)', txt)
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
    input_nodes = int(params['input nodes'])
    gradient_directions = int(params['gradient directions'])
    gradient_directions = min(gradient_directions, input_nodes-1)
    params2 = PARAMS0
    params2['network'] = {
        **PARAMS0['network'],
        'distribution samples': int(params['distribution samples']),
        'gradient directions': gradient_directions,
        'data for gradient': int(params['data for gradient']),
        'data for intercept': int(params['data for intercept']),
        'iteration layers': int(params['iteration layers']),
        'extra data filters': params['extra data filters']
    }
    params2['training'] = {
        **PARAMS0['training'],
        'input nodes': input_nodes,
        'fraction divisions tried': params['fraction divisions tried'],
        'minimum input node correlation': params['minimum input node correlation'],
        'maximum input node intercorrelation': params['maximum input node intercorrelation']
    }
    evaluation_function.ITERATION += 1
    print(f'---------- Iteration: {evaluation_function.ITERATION} ----------')
    print_json(params2, dest=ALCX_JSONFILE)
    print_json(params2, dest='stdout')

    stdout, stderr = run_job(ALCX_EXECUTABLE)
    if ALCX_OUTPUTLOG is not None:
        with open(ALCX_OUTPUTLOG, 'a') as f:
            f.write((f'---------- Iteration: {evaluation_function.ITERATION} ----------'))
            f.write(stdout)
    result = get_coefficient_of_determination(stdout)
    if result is not None:
        print(f"Result: {result}")
        value = -result['coeff']
        uncertainty = result['uncertainty']
    else:
        print(f"Result: {result}")
        value = 0.0
        uncertainty = 0.0

    param_df = dict_to_dataframe(params2)
    param_df[('results', 'coefficient of determination')] = -value
    param_df[('results', 'uncertainty')] = uncertainty
    if os.path.isfile(PARAMETER_HISTORY_FILE):
        param_df.to_csv(PARAMETER_HISTORY_FILE, mode='a', header=False)
    else:
        param_df.to_csv(PARAMETER_HISTORY_FILE)

    return value


def trials2df(result, trials):
    try:
        row1 = pd.DataFrame(trials.idxs_vals[1])
    except Exception as e:
        print(f'Exception: {e}')
        print('----- trials -----')
        print(trials)
        print('----- trials.idxs_vals -----')
        print(trials.idxs_vals)

    try:
        row2 = pd.DataFrame(trials.results)
    except Exception as e:
        print(f'Exception: {e}')
        print('----- trials -----')
        print(trials)
        print('----- trials.results -----')
        print(trials.results)

    return pd.concat((row1, row2), axis=1).reset_index(drop=True)


def hyperopt_optimize(opt_function, space, algo,
                      n_evals, random_state=None):
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
        max_evals=n_evals,
        rstate=np.random.RandomState(seed=random_state)
    )
    return {
        'result': result,
        'trials': trials
    }


def load_data():
    print('----- Loading data -----')
    import pickle
    with open(RESULT_FILE, 'rb') as f:
        result = pickle.load(f)
    with open(TRIAL_FILE, 'rb') as f:
        trials = pickle.load(f)
    return result, trials


def save_data(result, trials):
    print('----- Saving data -----')
    import pickle
    with open(RESULT_FILE, 'wb') as f:
        pickle.dump(result, f)
    with open(TRIAL_FILE, 'wb') as f:
        pickle.dump(trials, f)


evaluation_function.ITERATION = 0

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--exec_path', default=ALCX_EXECUTABLE)
    parser.add_argument('--run_folder', default=ALCX_RUN_FOLDER)
    parser.add_argument('--algorithm', default='rand')
    parser.add_argument('--random_state', default=12345, type=int)
    parser.add_argument('--evaluations', default=100, type=int)
    args = parser.parse_args()
    ALCX_EXECUTABLE = args.exec_path
    ALCX_RUN_FOLDER = args.run_folder

    if (ALCX_RUN_FOLDER is None) or (ALCX_EXECUTABLE is None):
        print('The environment variables ALCX_RUN_FOLDER and ALCX_EXECUTABLE')
        print('need to be defined.')

    ALCX_OUTPUTLOG = os.path.join(ALCX_RUN_FOLDER, f'alcx-{DATETIME_STAMP}.out')
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

    # ----- Preparation -----
    try:
        os.chdir(ALCX_RUN_FOLDER)
    except Exception as e:
        print(f"Unable to change folder to {ALCX_RUN_FOLDER}")
        print(f"Exception: {e}")
        quit()

    algo = args.algorithm.lower()
    if algo == 'tpe':
        algo = hyperopt.tpe.suggest
        PARAMETER_HISTORY_FILE = 'parameter_history__tpe_{DATETIME_STAMP}.csv'
    elif algo[:4] == 'rand':
        algo = hyperopt.rand.suggest
        PARAMETER_HISTORY_FILE = 'parameter_history_rand_{DATETIME_STAMP}.csv'
    else:
        print(f'Unknown algorithm: {algo}')
        quit()

    # ----- Optimize -----
    try:
        # result = evaluation_function(params)
        # print(f'Result: {result}')
        result = hyperopt_optimize(
            evaluation_function, space,
            algo, args.evaluations, int(args.random_state))
    except Exception as e:
        print('Problem encountered during hyperopt_optimize()')
        print(f"Exception: {e}")
        quit()

    # ----- Save results -----
    print('----- Hyperopt Optimization Completed -----')
    print(f"Result = {result['result']}")
    save_data(result, trials)
    df = trials2df(result['result'], result['trials'])
    df.to_csv(RESULTS_CSV_FILE)

    # ----- Best coefficient of determination -----
    df.reset_index(drop=True)
    best_coeff = df['coefficient of determination'].min()
    idxmin = df['coefficient of determination'].idxmin()
    print(f"Best coefficient of determination = {best_coeff} (at idx={idxmin})")
    print(f"Best set of parameters: {df.loc[idxmin, :]}")
