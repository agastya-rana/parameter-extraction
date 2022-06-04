import sys
sys.path.append('..')
sys.path.append('../src')
import numpy as np
import itertools
from src.local_methods import def_data_dir
from src.models import *
from src.plot_data import parse_population_params
main_dir = def_data_dir()

num_cells = 29
model = MWC_MM_Swayam()
#batches = [220517, 220518]
batches = [220517]
cells = [i for i in range(num_cells)]
stim_prot = "BS"
cell_names = ["%s_%s" % (batch, cell) for batch, cell in itertools.product(batches, cells)]

parse_population_params(model, stim_prot, cell_names, KDE=True)