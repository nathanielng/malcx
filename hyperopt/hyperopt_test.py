#!/usr/bin/env python

import numpy as np

from hyperopt_ex import hyperopt_optimize, fn_to_optimize


def test_hyperopt_optimize():
    my_func = lambda x: np.poly1d([2, -3, 4])(x)
    x = np.linspace(0, 2, 500)
    y = my_func(x)
    result = hyperopt_optimize(
        my_func, x.min(), x.max())
    assert np.abs( result['result']['x'] - 0.75 ) < 0.01
