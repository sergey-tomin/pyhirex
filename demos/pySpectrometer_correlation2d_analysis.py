# -*- coding: utf-8 -*-
"""
Svitozar Serkez

This is a temporary script file.
"""
import numpy as np
import matplotlib.pyplot as plt

#%%


# filepath = r'd:\DESYcloud\_code\pyhirex\test_data_onfly.npz'
# filepath = r"D:\DESYcloud\projects\2019_09_HXRSS_XFEL\pySpectrometer_scans\pitch scan 9100 full.npz"
# filepath = r"D:\DESYcloud\projects\2019_09_HXRSS_XFEL\pySpectrometer_scans\pitch scan 9100 full.npz"
# filepath = r'D:\DESYcloud\projects\2019_09_HXRSS_XFEL\pySpectrometer_scans\20200927-21_08_12_cor2d.npz'
# filepath = r'D:\DESYcloud\projects\2019_09_HXRSS_XFEL\pySpectrometer_scans\20200927-23_12_52_cor2d.npz'
filepath = r'D:\DESYcloud\projects\2019_09_HXRSS_XFEL\measured\all\20210427-03_10_57_cor2d.npz'
filepath = r'D:\DESYcloud\projects\2019_09_HXRSS_XFEL\measured\all\20210307-16_37_01_cor2d.npz'
tt = np.load(filepath)
print(tt.files)

corr2d = tt['corr2d']
spec_hist = tt['spec_hist']
doocs_scale = tt['doocs_scale']
phen_scale = tt['phen_scale']
doocs_vals_hist = tt['doocs_vals_hist']
doocs_label = tt['doocs_channel']


phen_scale = np.append(phen_scale,phen_scale[-1]+(phen_scale[-1]-phen_scale[-2])) # adding one more point so that matplotlib does not curse
#plotting the same plot as in XFELelog
plt.figure(345)
plt.clf()
plt.pcolormesh(doocs_scale, phen_scale, corr2d.T, shading='auto')
plt.xlabel(doocs_label)
plt.ylabel(r'$E_{ph}$ [eV]')
plt.show()

#plotting versus event number (here I'm skipping to speed-up plotting)
plt.figure(346)
plt.clf()
plt.pcolormesh(spec_hist.T[:,::10],shading='auto')
plt.ylabel(r'$E_{ph}$ [eV]')
plt.xlabel('event')
plt.show()

#you can bin the stuff yourself





# plt.figure(347)
# plt.clf()
# plt.plot(np.max(spec_hist.T,axis=0))
# plt.show()