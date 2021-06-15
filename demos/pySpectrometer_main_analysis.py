# -*- coding: utf-8 -*-
"""
Svitozar Serkez

This is a temporary script file.
"""
import numpy as np
import matplotlib.pyplot as plt

#%%

plt.close('all')

filepath = r'D:\DESYcloud\projects\2020_pyHirex_spectra\spectrum_14_09_2020_SASE_aftershift_tap40at24.npz'


tt = np.load(filepath)
print('In spectrum dump file \n {} there are following fields:\n {}'.format(filepath, str(tt.files)))
phen_scale = tt['e_axis']
spec_hist = tt['map']

event_scale_plus = np.arange(spec_hist.shape[1]+1)
phen_scale_plus = np.append(phen_scale,phen_scale[-1]+(phen_scale[-1]-phen_scale[-2])) # adding one more point so that matplotlib does not curse
plt.figure(1)
#plotting the same history plot as in XFELelog
plt.clf()
plt.pcolormesh(event_scale_plus, phen_scale_plus, spec_hist, shading='auto')
plt.ylabel(r'$E_{ph}$ [eV]')
plt.show()

plt.figure(2)
plt.clf()
plt.fill_between(phen_scale, np.amin(spec_hist, axis=1), np.amax(spec_hist, axis=1),color=[0.9,0.9,0.9])
plt.plot(phen_scale, spec_hist[:,0],color=[0.5,0.5,0.5])
plt.plot(phen_scale, np.mean(spec_hist, axis=1), color=[0.5,0,0])
plt.show()
