#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug  9 17:33:30 2020

@author: Sergey Tomin
"""

import numpy as np
import matplotlib.pyplot as plt

filename = "/home/xfeloper/user/pySpectrometer/SASE2/20200927-10_28_44_waterflow.npz"

data = np.load(filename)
e_axis = data["e_axis"] 
average = data["average"]
map2D = data["map"]

plt.figure(1)
plt.plot(e_axis, average)
plt.xlabel("Eph [eV]")
plt.ylabel("a.u.")

plt.figure(2)
plt.title("MAP")
plt.imshow(map2D)


fig, ax = plt.subplots(figsize=(7, 4))

N = 10 
for i in range(N):
    ax.plot(e_axis, map2D[:, i])
plt.title(f"First {N} spectrums")
plt.xlabel("Eph [eV]")
plt.ylabel("a.u.")
plt.show()