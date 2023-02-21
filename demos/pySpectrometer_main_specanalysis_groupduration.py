# -*- coding: utf-8 -*-
"""
Svitozar Serkez

This is a temporary script file.
"""
import numpy as np

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from copy import deepcopy

import matplotlib.pylab as pylab
params = {'legend.fontsize': 'large',
          #'figure.figsize': (15, 5),
         'axes.labelsize': 'large',
         'axes.titlesize':'large',
         'xtick.labelsize':'large',
         'ytick.labelsize':'large'}
pylab.rcParams.update(params)

import sys, os
sys.path.append('d:\DESYcloud\_code\pyhirex')
from opt_lib import *

E_min=0
E_max=np.inf
#%%

plt.close('all')

filepath = r'd:\DESYcloud\projects\2015_11_26_SASE3_two-color\2-color_plots_stability\20210520-01_44_32_specanalysis.npz'

dE = 0.5
E_min=700
E_max=705


tt = np.load(filepath)
print('In spectrum dump file \n {} there are following fields:\n {}'.format(filepath, str(tt.files)))
phen_scale = tt['phen_scale']
spec_hist = tt['spec_hist']

def delete_duplicate_events(spec_hist):
    spec0=spec_hist[:,0]*2
    spec_hist_new=[]
    for spec in spec_hist.T:
        if not np.allclose(spec,spec0):
            spec_hist_new.append(spec)
        spec0=spec
    return np.array(spec_hist_new).T

spec_hist = delete_duplicate_events(spec_hist)
#def delete_duplicate_events:




from opt_lib import *
spar = SpectrumArray()
spar.spec = spec_hist
spar.omega = phen_scale / hr_eV_s
spar.plot_lines(fignum=5)
spar.cut_spec(E_min,E_max)




corrn = spar.correlate_center(dE=dE, norm=1)
corrn.bin_phen(dE)
g2_fit = corrn.fit_g2func(g2_gauss, thresh=0)


E_ph_max = spar.phen[np.mean(spar.spec,axis=1).argmax()]

fit_contrast = np.array(g2_fit.fit_contrast)
fit_pedestal = np.array(g2_fit.fit_pedestal)
mult = fit_contrast / fit_pedestal
g2_fit.fit_t /= mult
g2_fit.plot_g2(phen=E_ph_max, plot_fit=1)#178 500
g2_fit.plot_t(spar=spar, fignum=7, thresh=0.3)


# g2_fit.plot_g2(phen=1000, plot_fit=1)#178 500
# g2_fit.plot_g2(phen=1005, plot_fit=1)#178 500
# g2_fit.plot_g2(phen=1010, plot_fit=1)#178 500
# g2_fit.plot_g2(phen=1015, plot_fit=1)#178 500