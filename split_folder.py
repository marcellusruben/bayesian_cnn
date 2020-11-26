# -*- coding: utf-8 -*-
"""

"""
import splitfolders 
import os

input_dir = os.path.join('C:/Users/Desktop/Bayesian CNN/flowers/')
output_dir = os.path.join('C:/Users/Desktop/Bayesian CNN/flowers_splitted/')

splitfolders.ratio(input_dir, output=output_dir, seed=1337, ratio=(.8, .2), group_prefix=None)