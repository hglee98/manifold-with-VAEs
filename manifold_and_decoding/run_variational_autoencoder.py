from __future__ import division
import numpy as np
import time
import datetime
import matplotlib.pyplot as plt
import os
import sys
gen_fn_dir = os.path.abspath('..') + '/shared_scripts'
sys.path.append(gen_fn_dir)

from dim_red_fns import run_vae
from binned_spikes_class import spike_counts
import general_file_fns as gff


sd = int((time.time() % 1)*(2**31))
np.random.seed(sd)
curr_date = datetime.datetime.now().strftime('%Y_%m_%d')+'_'

gen_params = gff.load_pickle_file('../general_params/general_params.p')


cols = gen_params['cols']
dir_to_save = gff.return_dir(gen_params['results_dir'] + 'dim_red_with_VAE/')

command_line = False
if command_line:
    session = sys.argv[1]
    state = sys.argv[2]
    # If condition is 'joint' should unpack state into first and second
    condition = sys.argv[3]
    target_dim = int(sys.argv[4])
    desired_nSamples = int(sys.argv[5])
else:
    session = 'Mouse12-120806'
    state = 'Wake'
    # state2 = 'REM'  # state2 is needed when the condition is 'joint'
    condition = 'solo'  # 'solo' or 'joint'
    target_dim = 3
# target dimension 은 3 또는 2
    desired_nSamples = 15000

print('Session %s, condition %s, target_dim %d, desired_nSamples %d' % (session, condition,
                                                                        target_dim, desired_nSamples))
area = 'ADn'
dt_kernel = 0.1  # subsampling interval (it should be a multiple of 0.05)
sigma = 0.1  # sigma is equal to 100ms
rate_params = {'dt': dt_kernel, 'sigma': sigma}
method = 'vae'
to_plot = True

t0 = time.time()

if condition == 'solo':

elif condition == 'joint':
