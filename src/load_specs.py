"""
Module for reading and parsing JSON specifications file for FRET data assimilation.

Agastya Rana, 11/12/2021.
"""

import os
from src.local_methods import def_data_dir
from src.single_cell_FRET import NP_PARAMS
import json
import numpy as np
data_dir = def_data_dir()
NP_arrs = NP_PARAMS

def read_specs_file(spec_name, data_dir=data_dir):
    """
    Function to read a specifications file.

    Module to gather information from specifications file about how a
    particular VA run is to be performed.
    Specs files should be JSON-readable, and should contain two dictionaries with keys of
    'data_vars', 'est_vars' (see README.md)

    At this point, there is no functional distinction between est_var
    and data_var, they are only tags for personal reference, and are
    read in as data types appropriate to the variable name.

    Args:
        spec_name: Name of specifications file.
        data_dir (object): Data folder, if different than in local_methods.

    Returns:
        list_dict: Dictionary of 2 items keyed by 'data_vars', 'est_vars', as should be specified in the specs file.
    """

    filename = '%s/specs/%s.txt' % (data_dir, spec_name)
    try:
        os.stat(filename)
    except:
        print("There is no input file %s/specs/%s.txt" % (data_dir, spec_name))
        exit()
    with open(filename, 'r') as specs_file:
        data = json.load(specs_file)
    print('\n -- Input vars and params loaded from %s.txt\n' % spec_name)

    out_dict = dict()
    out_dict.update(data['data_vars'])
    out_dict.update(data['est_vars'])
    for k in NP_arrs:
        if k in out_dict.keys():
            out_dict[k] = np.asarray(out_dict[k])
    return out_dict