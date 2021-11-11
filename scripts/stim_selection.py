## Imports

import sys, os
sys.path.append('D:\Yale\Emonet Lab\Fall 2021\FRET-data-assimilation')
## TODO: FIND A BETTER WAY TO DO THIS
## Problem is that just using current path location will only work if __name__ == __main__
#scripts_path = [i for i in sys.path if 'scripts' in i][0]
#sys.path.append(os.path.join(os.path.dirname(scripts_path),'src'))

from src.utils import get_flag
from scripts.est_VA import est_VA
from src.load_specs import read_specs_file, compile_all_run_vars
from src.single_cell_FRET import single_cell_FRET
import numpy as np
from src.local_methods import def_data_dir
main_dir = def_data_dir()

def try_stimulus(spec_name):
    # Load specifications from file; to be passed to single_cell_FRET object
    list_dict = read_specs_file(spec_name)
    vars_to_pass = compile_all_run_vars(list_dict)
    scF = single_cell_FRET(**vars_to_pass)

    scF.set_stim()
    scF.gen_true_states()
    scF.set_meas_data() ## Need to set noise  to 0 in specs file; note when adding noise, MWC_MM is scaled, MWC_linear not
    print("Here")
    param_set = est_VA(spec_name, scF=scF)
    print("Now now Here")
    return param_set

trials = 5
## Taken from Kamino (2020)
N_m = 2.018
N_sig = 0.387
## Below distribution is arbitrary
a_SS_m = 0.33
a_SS_sig = 0.10
slope_m = -0.01
slope_sig = 0.004

## Create parameter, stimulus inputs
N = np.random.lognormal(N_m, N_sig, trials)
a_SS = np.random.normal(a_SS_m, a_SS_sig, trials)
slope = np.random.normal(slope_m, slope_sig, trials)
tss = np.linspace(1, 20, 5)
l1s = np.linspace(0, 0.100, 5) ## Delta L in uM
avg_errors = np.zeros((len(tss)*len(l1s),3))
print(N[0], a_SS[0], slope[0])
j = 0
for ts, l1 in [(ts,l1) for ts in tss for l1 in l1s]:
    errors = np.zeros((trials))
    for i in range(trials):
        ## Create specs file for each input
        sp_name = '%s_%s_%s' % (ts, l1, i)
        filename = '%s/specs/%s.txt' % (main_dir, sp_name)
        #shutil.copyfile('%s/specs/template.txt' % main_dir, filename)
        with open('%s/specs/template.txt' % main_dir, 'r') as temp, open(filename, 'w') as f:
            for line in temp:
                if line.strip():
                    p = line.split()
                    if p[1] == "stim_params":
                        p[2] = "[%s,%s]" % (ts,l1)
                    elif p[1] == "params_set":
                        p[2] = "[20.,3225.,0.5,2.0,%s,%s,%s]" % (N[i], a_SS[i], slope[i])
                    line = " ".join([str(x) for x in p])
                f.write(line)
                f.write('\n')
        ## Run try_stimulus
        pset = try_stimulus(sp_name)
        print(pset)
        errors[i] = ((pset[-3]-N[i])/N[i])**2 + ((pset[-2]-a_SS[i])/a_SS[i])**2 + ((pset[-1]-slope[i])/slope[i])**2
    ## Average error over different parameter values
    ## Save results of t_s, l1, avg_error
    avg_errors[j] = [ts, l1, np.average(errors)]
    j = j + 1

np.save('%s/results/results.npy' % main_dir, avg_errors)