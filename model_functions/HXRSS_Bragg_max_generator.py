"""
Created on Mon Nov 11 14:12:59 2019

@author: ggeloni
Edited to function by cgrech (Jul 6, 21)
"""

from sympy.utilities.iterables import multiset_permutations
import sys
import os
import matplotlib.pyplot as plt
import numpy as np
import time
import logging


def HXRSS_Bragg_max_generator(thplist, h_max, k_max, l_max, dthp, dthy, roll_angle_list, dthr, alpha):
    p_angle_list = []
    phen_list = []
    label_list = []
    linestyle_list = []
    gid_list = []
    color_list = []
    roll_list = []

    #annotaz = plot.annotate(" ", (0,0), (-60, 650), xycoords='axes fraction', textcoords='offset points', va='top')

    def rotm(th, ux, uy, uz):
        r = np.array((
            (ux*ux*(1-np.cos(th))+np.cos(th),     ux*uy*(1-np.cos(th))
             - uz*np.sin(th),     ux*uz*(1-np.cos(th))+uy*np.sin(th)),
            (ux*uy*(1-np.cos(th))+uz*np.sin(th),  uy*uy*(1-np.cos(th))
             + np.cos(th),        uy*uz*(1-np.cos(th))-ux*np.sin(th)),
            (ux*uz*(1-np.cos(th))-uy*np.sin(th),  uy*uz*(1-np.cos(th))
             + ux*np.sin(th),     uz*uz*(1-np.cos(th))+np.cos(th))
            ))
        return r

    #(1) Pitch of thp around PitchAx, and rotation of Yaw and Roll axis:
    def rotm1(thp, pitchax, rollax, yawax):
        r1 = rotm(np.pi/2-thp, pitchax[0], pitchax[1], pitchax[2])
        rollax2 = r1.dot(rollax)
        yawax2 = r1.dot(yawax)
        return r1, rollax2, yawax2

    #(2) Yaw of thy around yawax2, and rotation of Roll axis:
    def rotm2(thy, rollax2, yawax2):
        r2 = rotm(thy, yawax2[0], yawax2[1], yawax2[2])
        rollax3 = r2.dot(rollax2)
        return r2, rollax3

    #(3) Roll of thr around Rollax3:
    def rotm3(thr, rollax3):
        r3 = rotm(thr, rollax3[0], rollax3[1], rollax3[2])
        return r3

    def kirot(thp, thy, thr, n0, pitchax, rollax, yawax):
        #note: it seems like in Alberto's tool the roll and yaw are not transformed by subsequent rotations. For comparison, Ileave this out too.
        r1, rollax2, yawax2 = rotm1(thp, pitchax, rollax, yawax)
        rollax2 = rollax
        yawax2 = yawax
        r2, rollax3 = rotm2(thy, rollax2, yawax2)
        rollax3 = rollax
        r3 = rotm3(thr, rollax3)
        return r3.dot(r2.dot(r1.dot(n0)))

    def phev(fact, n, h, k, l, a, thp, thy, thr, n0, pitchax, rollax, yawax):
        d = a/np.sqrt(h**2+k**2+l**2)
        return fact*np.sqrt(h**2+k**2+l**2)/(2*d*n*np.linalg.norm(kirot(thp, thy, thr, n0, pitchax, rollax, yawax).dot((h, k, l))))

    def plotene(thplist, fact, n, h, k, l, a, DTHP, thylist, thr, n0, pitchax, rollax, yawax):
        count = 0
        for thp, thy in zip(thplist/180*np.pi, thylist):
            eevlist[count] = (phev(fact, 1, h, k, l, a, thp,
                                   thy, thr, n0, pitchax, rollax, yawax))
            count = count+1
        rosso = np.array((1, 1, 1))
        verde = np.array((2, 2, 0))
        nero = np.array((1, 1, 3))
        blu = np.array((4, 0, 0))
        magenta = np.array((3, 3, 1))
        arancio = np.array((2, 2, 4))
        azzurro = np.array((3, 3, 3))
        giallo = np.array((1, 1, 5))
        viola = np.array((1, 3, 5))
        marrone = np.array((5, 5, 5))
        porpora = np.array((1, 5, 5))
        grigio = np.array((3, 5, 5))
        oro = np.array((3, 5, 5))
        beige = np.array((3, 3, 5))
        aquamarine = np.array((4, 4, 4))
        wheat = np.array((4, 4, 0))
        if [np.abs(h), np.abs(k), np.abs(l)] in list(multiset_permutations(rosso)):
            colore = 'r'
        elif [np.abs(h), np.abs(k), np.abs(l)] in list(multiset_permutations(verde)):
            colore = 'g'
        elif [np.abs(h), np.abs(k), np.abs(l)] in list(multiset_permutations(nero)):
            colore = 'k'
        elif [np.abs(h), np.abs(k), np.abs(l)] in list(multiset_permutations(blu)):
            colore = 'b'
        elif [np.abs(h), np.abs(k), np.abs(l)] in list(multiset_permutations(magenta)):
            colore = 'm'
        elif [np.abs(h), np.abs(k), np.abs(l)] in list(multiset_permutations(arancio)):
            colore = 'y'
        elif [np.abs(h), np.abs(k), np.abs(l)] in list(multiset_permutations(azzurro)):
            colore = 'c'
        elif [np.abs(h), np.abs(k), np.abs(l)] in list(multiset_permutations(giallo)):
            colore = 'r'
        elif [np.abs(h), np.abs(k), np.abs(l)] in list(multiset_permutations(viola)):
            colore = 'g'
        elif [np.abs(h), np.abs(k), np.abs(l)] in list(multiset_permutations(marrone)):
            colore = 'k'
        elif [np.abs(h), np.abs(k), np.abs(l)] in list(multiset_permutations(porpora)):
            colore = 'b'
        elif [np.abs(h), np.abs(k), np.abs(l)] in list(multiset_permutations(grigio)):
            colore = 'm'
        elif [np.abs(h), np.abs(k), np.abs(l)] in list(multiset_permutations(oro)):
            colore = 'y'
        elif [np.abs(h), np.abs(k), np.abs(l)] in list(multiset_permutations(beige)):
            colore = 'c'
        elif [np.abs(h), np.abs(k), np.abs(l)] in list(multiset_permutations(aquamarine)):
            colore = 'r'
        elif [np.abs(h), np.abs(k), np.abs(l)] in list(multiset_permutations(wheat)):
            colore = 'g'
        else:
            colore = 'k'
        simbolo = 'dashed'
        if (h >= 0 and k >= 0 and l >= 0) or (h <= 0 and k <= 0 and l <= 0):
            simbolo = 'solid'
        if not(h == k):
            simbolo = 'dashdot'
        gid = [h, k, l]
        thplist_f = thplist+DTHP
        color = colore
        linestyle = str(simbolo)
        return thplist_f, eevlist, gid, linestyle, color

#User defined quantities
################ AMERICAN NAME CONVENTION --- OUR ROLL IS YAW HERE AND VICEVERSA!!!######################

    #pitchax = np.array((-1, 1, 0))/np.linalg.norm(np.array((-1, 1, 0))) Changed convention
    pitchax = np.array((1, -1, 0))/np.linalg.norm(np.array((1, -1, 0)))

    rollax = np.array((0, 0, 1))/np.linalg.norm(np.array((0, 0, 1)))
    yawax = np.array((1, 1, 0))/np.linalg.norm(np.array((1, 1, 0)))
    n0 = -rollax  # direction of incident radiation

    a = 3.5667899884942195e-10
    hbar = 1.05457173e-34
    clight = 299792458.0
    eel = 1.60217657e-19

    fact = 2*np.pi*clight*hbar/eel

    nord = 1
    hmax = h_max
    kmax = k_max
    lmax = l_max

    eevlist = np.zeros(len(thplist))

    DTHP = dthp  # -0.6921-0.09

    for roll_angle in roll_angle_list:
        DTHY = dthy+(alpha*thplist)  # 15#15#0#-0.15#-0.39#-0.15 #0.0885
        DTHR = dthr
        # AMERICAN YAW DEFINITION, our roll angle
        thylist = (-DTHY+roll_angle)/180*np.pi
        thr = (-DTHR)/180*np.pi  # AMERICAN ROLL DEFINITION

        for h in range(0, hmax+1):
            for k in range(-kmax, kmax+1):
               for l in range(-lmax, lmax+1):
                   ref = h*np.array((1, 0, 0))+k * \
                                    np.array((0, 1, 0))+l*np.array((0, 0, 1))
                   allowed = 0
                   if (h % 2 and k % 2 and l % 2) or (not(h % 2) and not(k % 2) and not(l % 2) and not((h+k+l) % 4)) and not(h == 0 and k == 0 and l == 0):
                       allowed = 1
                       if h == 0 and k < 0:
                           allowed = 0
                       if h == 0 and k == 0 and l < 0:
                           allowed = 0
                       if allowed == 1:
                           p_angle, phen, gid, linestyle, color = plotene(
                               thplist, fact, nord, h, k, l, a, DTHP, thylist, thr, n0, pitchax, rollax, yawax)
                           phen_list.append(list(phen))
                           p_angle_list.append(list(p_angle))
                           gid_list.append(str(gid))
                           color_list.append(str(color))
                           linestyle_list.append(str(linestyle))
                           roll_list.append(roll_angle)
    return phen_list, p_angle_list, gid_list, roll_list, color_list, linestyle_list
