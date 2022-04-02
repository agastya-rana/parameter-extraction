"""
Module for reading and parsing JSON specifications file for FRET data assimilation.

Agastya Rana, 11/12/2021.
"""

import os
from local_methods import def_data_dir
import json

data_dir = def_data_dir()

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
    return data


def compile_all_run_vars(list_dict):
    """
    Grab all the run variables from the specifications file, and aggregate
    as a complete dictionary of variables to be specified and/or overriden
    in the single_cell_FRET_VA object.

    Args:
        list_dict: dictionary containing at least 2 keys; data_vars and
            est_vars. These are read through read_specs_file()
            function in this module. Only these two keys are read
            in this module; other keys may exist, but will be ignored.

    Returns:
        vars_to_pass: dictionary whose keys are all variables to be overriden
        in the single_cell_FRET class when initialized.
    """

    vars_to_pass = dict()
    vars_to_pass.update(list_dict['data_vars'])
    vars_to_pass.update(list_dict['est_vars'])
    return vars_to_pass
