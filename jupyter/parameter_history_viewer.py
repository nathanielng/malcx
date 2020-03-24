#!/usr/bin/env python
# coding: utf-8

# In[1]:


import glob
import matplotlib.pyplot as plt
import os
import pandas as pd
import sys

sys.path.insert(0, os.path.abspath('../src'))

from alcx_plot import add_timestamps, summarize_historyfile
from alcx_plot import plot_parameter_histograms, plot_parameter_history, plot_parameter_correlation
from pandas.plotting import scatter_matrix


# In[2]:


RUN_FOLDER = os.path.abspath('../../run/')

files = glob.glob(f'{RUN_FOLDER}/parameter_history*.csv')
file_list = add_timestamps(files)
last_file = file_list.iloc[-3].loc['Filename']
print(f'Loading last parameter history file:\n{last_file}')

df = summarize_historyfile(last_file)


# In[3]:


plot_parameter_histograms(df)


# In[4]:


plot_parameter_history(df)


# In[5]:


plot_parameter_correlation(df)


# In[6]:


scatter_matrix(df, figsize=(20,20));

