#!/usr/bin/env python

import hyperopt
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def fn_to_optimize(x):
    return np.poly1d([2, -3, 4])(x)


def plot_fn(x, y, filename):
    miny = min(y)
    maxy = max(y)
    minx = x[np.argmin(y)]

    fig, ax = plt.subplots(1, 1, figsize=(7,5))
    ax.plot(x, y)
    ax.vlines(minx, miny-0.25*(maxy-miny), maxy, ls='--', colors='gray')
    ax.plot([minx], [miny], 'ro', markersize=15,
        markerfacecolor='None', markeredgewidth=2)
    ax.annotate(f"({minx:.4f}, {miny:.4f})   ",
                xy=(minx,miny), ha='right', va='top')
    plt.savefig(filename)


def hyperopt_optimize(my_fn, x_min, x_max,
                      algo=hyperopt.rand.suggest):
    trials = hyperopt.Trials()
    result = hyperopt.fmin(
        fn=my_fn,
        space=hyperopt.hp.uniform('x', x_min, x_max),
        algo=algo,
        trials=trials,
        max_evals=100,
        rstate=np.random.RandomState(100)
    )
    return {
        'result': result,
        'trials': trials
    }


def trials2df(result, trials):
    x = pd.Series(trials.idxs_vals[1]['x'])
    loss = pd.Series([x['loss'] for x in trials.results])
    df = pd.DataFrame({
        'x': x,
        'loss': loss
        }, index = trials.idxs_vals[0]['x'])
    return df


if __name__ == "__main__":
    x = np.linspace(0, 2, 500)
    y = fn_to_optimize(x)
    plot_fn(x, y, 'hyperopt.png')
    result = hyperopt_optimize(fn_to_optimize, x.min(), x.max())
    print(f"Result = {result['result']}")
    df = trials2df(result['result'], result['trials'])
    print(df.head())
    print(df.tail())
    print()
    idxmin = df['loss'].idxmin()
    print(f"Best x: {df.loc[idxmin, 'x']}")
    print(f"with loss = {df['loss'].min()} (at iteration # {idxmin})")
