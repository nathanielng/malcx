#!/bin/bash

for file in .gitignore README.md
do
    echo $file
    chmod -x $file
done


pushd hyperopt
for file in hyperopt_dist.py
do
    echo $file
    chmod -x $file
done
popd


pushd jupyter
for file in parameter_history_viewer.*
do
    echo $file
    chmod -x $file
done
popd


pushd src
for file in alcx*.py run.sh
do
    echo $file
    chmod -x $file
done
popd