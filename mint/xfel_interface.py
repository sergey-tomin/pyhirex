"""
XFEL machine interface
S.Tomin, 2017
"""
from __future__ import absolute_import, print_function

try:
    # in server "doocsdev12" set environment
    #  $ export PYTHONPATH=/home/ttflinac/user/python-2.7/Debian/
    import pydoocs
except:
    pass # Show message on Constructor if we try to use it.

import os
import sys
import numpy as np
import subprocess
import base64
from mint.opt_objects import MachineInterface, Device, TestDevice
from collections import OrderedDict
from datetime import datetime
import json

class AlarmDevice(Device):
    """
    Devices for getting information about Machine status
    """
    def __init__(self, eid=None):
        super(AlarmDevice, self).__init__(eid=eid)

machine_readout_list = ["XFEL.FEL/WAVELENGTHCONTROL.SA1/U40.2250.SA1/KVAL.PREDICT",
"XFEL.FEL/WAVELENGTHCONTROL.SA1/U40.2256.SA1/KVAL.PREDICT",
"XFEL.FEL/WAVELENGTHCONTROL.SA1/U40.2262.SA1/KVAL.PREDICT",
"XFEL.FEL/WAVELENGTHCONTROL.SA1/U40.2269.SA1/KVAL.PREDICT",
"XFEL.FEL/WAVELENGTHCONTROL.SA1/U40.2275.SA1/KVAL.PREDICT",
"XFEL.FEL/WAVELENGTHCONTROL.SA1/U40.2281.SA1/KVAL.PREDICT",
"XFEL.FEL/WAVELENGTHCONTROL.SA1/U40.2287.SA1/KVAL.PREDICT",
"XFEL.FEL/WAVELENGTHCONTROL.SA1/U40.2293.SA1/KVAL.PREDICT",
"XFEL.FEL/WAVELENGTHCONTROL.SA1/U40.2299.SA1/KVAL.PREDICT",
"XFEL.FEL/WAVELENGTHCONTROL.SA1/U40.2305.SA1/KVAL.PREDICT",
"XFEL.FEL/WAVELENGTHCONTROL.SA1/U40.2311.SA1/KVAL.PREDICT",
"XFEL.FEL/WAVELENGTHCONTROL.SA1/U40.2317.SA1/KVAL.PREDICT",
"XFEL.FEL/WAVELENGTHCONTROL.SA1/U40.2323.SA1/KVAL.PREDICT",
"XFEL.FEL/WAVELENGTHCONTROL.SA1/U40.2330.SA1/KVAL.PREDICT",
"XFEL.FEL/WAVELENGTHCONTROL.SA1/U40.2336.SA1/KVAL.PREDICT",
"XFEL.FEL/WAVELENGTHCONTROL.SA1/U40.2342.SA1/KVAL.PREDICT",
"XFEL.FEL/WAVELENGTHCONTROL.SA1/U40.2348.SA1/KVAL.PREDICT",
"XFEL.FEL/WAVELENGTHCONTROL.SA1/U40.2354.SA1/KVAL.PREDICT",
"XFEL.FEL/WAVELENGTHCONTROL.SA1/U40.2360.SA1/KVAL.PREDICT",
"XFEL.FEL/WAVELENGTHCONTROL.SA1/U40.2366.SA1/KVAL.PREDICT",
"XFEL.FEL/WAVELENGTHCONTROL.SA1/U40.2372.SA1/KVAL.PREDICT",
"XFEL.FEL/WAVELENGTHCONTROL.SA1/U40.2378.SA1/KVAL.PREDICT",
"XFEL.FEL/WAVELENGTHCONTROL.SA1/U40.2384.SA1/KVAL.PREDICT",
"XFEL.FEL/WAVELENGTHCONTROL.SA1/U40.2391.SA1/KVAL.PREDICT",
"XFEL.FEL/WAVELENGTHCONTROL.SA1/U40.2397.SA1/KVAL.PREDICT",
"XFEL.FEL/WAVELENGTHCONTROL.SA1/U40.2403.SA1/KVAL.PREDICT",
"XFEL.FEL/WAVELENGTHCONTROL.SA1/U40.2409.SA1/KVAL.PREDICT",
"XFEL.FEL/WAVELENGTHCONTROL.SA1/U40.2415.SA1/KVAL.PREDICT",
"XFEL.FEL/WAVELENGTHCONTROL.SA1/U40.2421.SA1/KVAL.PREDICT",
"XFEL.FEL/WAVELENGTHCONTROL.SA1/U40.2427.SA1/KVAL.PREDICT",
"XFEL.FEL/WAVELENGTHCONTROL.SA1/U40.2433.SA1/KVAL.PREDICT",
"XFEL.FEL/WAVELENGTHCONTROL.SA1/U40.2439.SA1/KVAL.PREDICT",
"XFEL.FEL/WAVELENGTHCONTROL.SA1/U40.2445.SA1/KVAL.PREDICT",
"XFEL.FEL/WAVELENGTHCONTROL.SA1/U40.2452.SA1/KVAL.PREDICT",
"XFEL.FEL/WAVELENGTHCONTROL.SA1/U40.2458.SA1/KVAL.PREDICT",
"XFEL.FEL/WAVELENGTHCONTROL.SA2/U40.2200.SA2/KVAL.PREDICT",
"XFEL.FEL/WAVELENGTHCONTROL.SA2/U40.2206.SA2/KVAL.PREDICT",
"XFEL.FEL/WAVELENGTHCONTROL.SA2/U40.2212.SA2/KVAL.PREDICT",
"XFEL.FEL/WAVELENGTHCONTROL.SA2/U40.2218.SA2/KVAL.PREDICT",
"XFEL.FEL/WAVELENGTHCONTROL.SA2/U40.2224.SA2/KVAL.PREDICT",
"XFEL.FEL/WAVELENGTHCONTROL.SA2/U40.2230.SA2/KVAL.PREDICT",
"XFEL.FEL/WAVELENGTHCONTROL.SA2/U40.2237.SA2/KVAL.PREDICT",
"XFEL.FEL/WAVELENGTHCONTROL.SA2/U40.2243.SA2/KVAL.PREDICT",
"XFEL.FEL/WAVELENGTHCONTROL.SA2/U40.2255.SA2/KVAL.PREDICT",
"XFEL.FEL/WAVELENGTHCONTROL.SA2/U40.2261.SA2/KVAL.PREDICT",
"XFEL.FEL/WAVELENGTHCONTROL.SA2/U40.2267.SA2/KVAL.PREDICT",
"XFEL.FEL/WAVELENGTHCONTROL.SA2/U40.2273.SA2/KVAL.PREDICT",
"XFEL.FEL/WAVELENGTHCONTROL.SA2/U40.2279.SA2/KVAL.PREDICT",
"XFEL.FEL/WAVELENGTHCONTROL.SA2/U40.2285.SA2/KVAL.PREDICT",
"XFEL.FEL/WAVELENGTHCONTROL.SA2/U40.2291.SA2/KVAL.PREDICT",
"XFEL.FEL/WAVELENGTHCONTROL.SA2/U40.2297.SA2/KVAL.PREDICT",
"XFEL.FEL/WAVELENGTHCONTROL.SA2/U40.2310.SA2/KVAL.PREDICT",
"XFEL.FEL/WAVELENGTHCONTROL.SA2/U40.2316.SA2/KVAL.PREDICT",
"XFEL.FEL/WAVELENGTHCONTROL.SA2/U40.2322.SA2/KVAL.PREDICT",
"XFEL.FEL/WAVELENGTHCONTROL.SA2/U40.2328.SA2/KVAL.PREDICT",
"XFEL.FEL/WAVELENGTHCONTROL.SA2/U40.2334.SA2/KVAL.PREDICT",
"XFEL.FEL/WAVELENGTHCONTROL.SA2/U40.2340.SA2/KVAL.PREDICT",
"XFEL.FEL/WAVELENGTHCONTROL.SA2/U40.2346.SA2/KVAL.PREDICT",
"XFEL.FEL/WAVELENGTHCONTROL.SA2/U40.2352.SA2/KVAL.PREDICT",
"XFEL.FEL/WAVELENGTHCONTROL.SA2/U40.2358.SA2/KVAL.PREDICT",
"XFEL.FEL/WAVELENGTHCONTROL.SA2/U40.2365.SA2/KVAL.PREDICT",
"XFEL.FEL/WAVELENGTHCONTROL.SA2/U40.2371.SA2/KVAL.PREDICT",
"XFEL.FEL/WAVELENGTHCONTROL.SA2/U40.2377.SA2/KVAL.PREDICT",
"XFEL.FEL/WAVELENGTHCONTROL.SA2/U40.2383.SA2/KVAL.PREDICT",
"XFEL.FEL/WAVELENGTHCONTROL.SA2/U40.2389.SA2/KVAL.PREDICT",
"XFEL.FEL/WAVELENGTHCONTROL.SA2/U40.2395.SA2/KVAL.PREDICT",
"XFEL.FEL/WAVELENGTHCONTROL.SA2/U40.2401.SA2/KVAL.PREDICT",
"XFEL.FEL/WAVELENGTHCONTROL.SA2/U40.2407.SA2/KVAL.PREDICT",
"XFEL.FEL/WAVELENGTHCONTROL.SA2/U40.2413.SA2/KVAL.PREDICT",
"XFEL.FEL/WAVELENGTHCONTROL.SA2/U40.2419.SA2/KVAL.PREDICT",
"XFEL.FEL/WAVELENGTHCONTROL.SA3/U68.2809.SA3/KVAL.PREDICT",
"XFEL.FEL/WAVELENGTHCONTROL.SA3/U68.2815.SA3/KVAL.PREDICT",
"XFEL.FEL/WAVELENGTHCONTROL.SA3/U68.2821.SA3/KVAL.PREDICT",
"XFEL.FEL/WAVELENGTHCONTROL.SA3/U68.2827.SA3/KVAL.PREDICT",
"XFEL.FEL/WAVELENGTHCONTROL.SA3/U68.2834.SA3/KVAL.PREDICT",
"XFEL.FEL/WAVELENGTHCONTROL.SA3/U68.2840.SA3/KVAL.PREDICT",
"XFEL.FEL/WAVELENGTHCONTROL.SA3/U68.2846.SA3/KVAL.PREDICT",
"XFEL.FEL/WAVELENGTHCONTROL.SA3/U68.2852.SA3/KVAL.PREDICT",
"XFEL.FEL/WAVELENGTHCONTROL.SA3/U68.2858.SA3/KVAL.PREDICT",
"XFEL.FEL/WAVELENGTHCONTROL.SA3/U68.2864.SA3/KVAL.PREDICT",
"XFEL.FEL/WAVELENGTHCONTROL.SA3/U68.2870.SA3/KVAL.PREDICT",
"XFEL.FEL/WAVELENGTHCONTROL.SA3/U68.2882.SA3/KVAL.PREDICT",
"XFEL.FEL/WAVELENGTHCONTROL.SA3/U68.2888.SA3/KVAL.PREDICT",
"XFEL.FEL/WAVELENGTHCONTROL.SA3/U68.2894.SA3/KVAL.PREDICT",
"XFEL.FEL/WAVELENGTHCONTROL.SA3/U68.2901.SA3/KVAL.PREDICT",
"XFEL.FEL/WAVELENGTHCONTROL.SA3/U68.2907.SA3/KVAL.PREDICT",
"XFEL.FEL/WAVELENGTHCONTROL.SA3/U68.2913.SA3/KVAL.PREDICT",
"XFEL.FEL/WAVELENGTHCONTROL.SA3/U68.2919.SA3/KVAL.PREDICT",
"XFEL.FEL/WAVELENGTHCONTROL.SA3/U68.2925.SA3/KVAL.PREDICT",
"XFEL.FEL/WAVELENGTHCONTROL.SA3/U68.2931.SA3/KVAL.PREDICT",
"XFEL.FEL/WAVELENGTHCONTROL.SA3/U68.2937.SA3/KVAL.PREDICT",
"XFEL.FEL/UNDULATOR.SASE1/U40.2250.SA1/GAP",
"XFEL.FEL/UNDULATOR.SASE1/BPS.2254.SA1/GAP",
"XFEL.FEL/UNDULATOR.SASE1/U40.2256.SA1/GAP",
"XFEL.FEL/UNDULATOR.SASE1/BPS.2260.SA1/GAP",
"XFEL.FEL/UNDULATOR.SASE1/U40.2262.SA1/GAP",
"XFEL.FEL/UNDULATOR.SASE1/BPS.2266.SA1/GAP",
"XFEL.FEL/UNDULATOR.SASE1/U40.2269.SA1/GAP",
"XFEL.FEL/UNDULATOR.SASE1/BPS.2272.SA1/GAP",
"XFEL.FEL/UNDULATOR.SASE1/U40.2275.SA1/GAP",
"XFEL.FEL/UNDULATOR.SASE1/BPS.2278.SA1/GAP",
"XFEL.FEL/UNDULATOR.SASE1/U40.2281.SA1/GAP",
"XFEL.FEL/UNDULATOR.SASE1/BPS.2284.SA1/GAP",
"XFEL.FEL/UNDULATOR.SASE1/U40.2287.SA1/GAP",
"XFEL.FEL/UNDULATOR.SASE1/BPS.2290.SA1/GAP",
"XFEL.FEL/UNDULATOR.SASE1/U40.2293.SA1/GAP",
"XFEL.FEL/UNDULATOR.SASE1/BPS.2296.SA1/GAP",
"XFEL.FEL/UNDULATOR.SASE1/U40.2299.SA1/GAP",
"XFEL.FEL/UNDULATOR.SASE1/BPS.2302.SA1/GAP",
"XFEL.FEL/UNDULATOR.SASE1/U40.2305.SA1/GAP",
"XFEL.FEL/UNDULATOR.SASE1/BPS.2309.SA1/GAP",
"XFEL.FEL/UNDULATOR.SASE1/U40.2311.SA1/GAP",
"XFEL.FEL/UNDULATOR.SASE1/BPS.2315.SA1/GAP",
"XFEL.FEL/UNDULATOR.SASE1/U40.2317.SA1/GAP",
"XFEL.FEL/UNDULATOR.SASE1/BPS.2321.SA1/GAP",
"XFEL.FEL/UNDULATOR.SASE1/U40.2323.SA1/GAP",
"XFEL.FEL/UNDULATOR.SASE1/BPS.2327.SA1/GAP",
"XFEL.FEL/UNDULATOR.SASE1/U40.2330.SA1/GAP",
"XFEL.FEL/UNDULATOR.SASE1/BPS.2333.SA1/GAP",
"XFEL.FEL/UNDULATOR.SASE1/U40.2336.SA1/GAP",
"XFEL.FEL/UNDULATOR.SASE1/BPS.2339.SA1/GAP",
"XFEL.FEL/UNDULATOR.SASE1/U40.2342.SA1/GAP",
"XFEL.FEL/UNDULATOR.SASE1/BPS.2345.SA1/GAP",
"XFEL.FEL/UNDULATOR.SASE1/U40.2348.SA1/GAP",
"XFEL.FEL/UNDULATOR.SASE1/BPS.2351.SA1/GAP",
"XFEL.FEL/UNDULATOR.SASE1/U40.2354.SA1/GAP",
"XFEL.FEL/UNDULATOR.SASE1/BPS.2357.SA1/GAP",
"XFEL.FEL/UNDULATOR.SASE1/U40.2360.SA1/GAP",
"XFEL.FEL/UNDULATOR.SASE1/BPS.2363.SA1/GAP",
"XFEL.FEL/UNDULATOR.SASE1/U40.2366.SA1/GAP",
"XFEL.FEL/UNDULATOR.SASE1/BPS.2370.SA1/GAP",
"XFEL.FEL/UNDULATOR.SASE1/U40.2372.SA1/GAP",
"XFEL.FEL/UNDULATOR.SASE1/BPS.2376.SA1/GAP",
"XFEL.FEL/UNDULATOR.SASE1/U40.2378.SA1/GAP",
"XFEL.FEL/UNDULATOR.SASE1/BPS.2382.SA1/GAP",
"XFEL.FEL/UNDULATOR.SASE1/U40.2384.SA1/GAP",
"XFEL.FEL/UNDULATOR.SASE1/BPS.2388.SA1/GAP",
"XFEL.FEL/UNDULATOR.SASE1/U40.2391.SA1/GAP",
"XFEL.FEL/UNDULATOR.SASE1/BPS.2394.SA1/GAP",
"XFEL.FEL/UNDULATOR.SASE1/U40.2397.SA1/GAP",
"XFEL.FEL/UNDULATOR.SASE1/BPS.2400.SA1/GAP",
"XFEL.FEL/UNDULATOR.SASE1/U40.2403.SA1/GAP",
"XFEL.FEL/UNDULATOR.SASE1/BPS.2406.SA1/GAP",
"XFEL.FEL/UNDULATOR.SASE1/U40.2409.SA1/GAP",
"XFEL.FEL/UNDULATOR.SASE1/BPS.2412.SA1/GAP",
"XFEL.FEL/UNDULATOR.SASE1/U40.2415.SA1/GAP",
"XFEL.FEL/UNDULATOR.SASE1/BPS.2418.SA1/GAP",
"XFEL.FEL/UNDULATOR.SASE1/U40.2421.SA1/GAP",
"XFEL.FEL/UNDULATOR.SASE1/BPS.2424.SA1/GAP",
"XFEL.FEL/UNDULATOR.SASE1/U40.2427.SA1/GAP",
"XFEL.FEL/UNDULATOR.SASE1/BPS.2431.SA1/GAP",
"XFEL.FEL/UNDULATOR.SASE1/U40.2433.SA1/GAP",
"XFEL.FEL/UNDULATOR.SASE1/BPS.2437.SA1/GAP",
"XFEL.FEL/UNDULATOR.SASE1/U40.2439.SA1/GAP",
"XFEL.FEL/UNDULATOR.SASE1/BPS.2443.SA1/GAP",
"XFEL.FEL/UNDULATOR.SASE1/U40.2445.SA1/GAP",
"XFEL.FEL/UNDULATOR.SASE1/BPS.2449.SA1/GAP",
"XFEL.FEL/UNDULATOR.SASE1/U40.2452.SA1/GAP",
"XFEL.FEL/UNDULATOR.SASE1/BPS.2455.SA1/GAP",
"XFEL.FEL/UNDULATOR.SASE1/U40.2458.SA1/GAP",
"XFEL.FEL/UNDULATOR.SASE2/U40.2200.SA2/GAP",
"XFEL.FEL/UNDULATOR.SASE2/BPS.2203.SA2/GAP",
"XFEL.FEL/UNDULATOR.SASE2/U40.2206.SA2/GAP",
"XFEL.FEL/UNDULATOR.SASE2/BPS.2209.SA2/GAP",
"XFEL.FEL/UNDULATOR.SASE2/U40.2212.SA2/GAP",
"XFEL.FEL/UNDULATOR.SASE2/BPS.2216.SA2/GAP",
"XFEL.FEL/UNDULATOR.SASE2/U40.2218.SA2/GAP",
"XFEL.FEL/UNDULATOR.SASE2/BPS.2222.SA2/GAP",
"XFEL.FEL/UNDULATOR.SASE2/U40.2224.SA2/GAP",
"XFEL.FEL/UNDULATOR.SASE2/BPS.2228.SA2/GAP",
"XFEL.FEL/UNDULATOR.SASE2/U40.2230.SA2/GAP",
"XFEL.FEL/UNDULATOR.SASE2/BPS.2234.SA2/GAP",
"XFEL.FEL/UNDULATOR.SASE2/U40.2237.SA2/GAP",
"XFEL.FEL/UNDULATOR.SASE2/BPS.2240.SA2/GAP",
"XFEL.FEL/UNDULATOR.SASE2/U40.2243.SA2/GAP",
"XFEL.FEL/UNDULATOR.SASE2/BPS.2252.SA2/GAP",
"XFEL.FEL/UNDULATOR.SASE2/U40.2255.SA2/GAP",
"XFEL.FEL/UNDULATOR.SASE2/BPS.2258.SA2/GAP",
"XFEL.FEL/UNDULATOR.SASE2/U40.2261.SA2/GAP",
"XFEL.FEL/UNDULATOR.SASE2/BPS.2264.SA2/GAP",
"XFEL.FEL/UNDULATOR.SASE2/U40.2267.SA2/GAP",
"XFEL.FEL/UNDULATOR.SASE2/BPS.2270.SA2/GAP",
"XFEL.FEL/UNDULATOR.SASE2/U40.2273.SA2/GAP",
"XFEL.FEL/UNDULATOR.SASE2/BPS.2276.SA2/GAP",
"XFEL.FEL/UNDULATOR.SASE2/U40.2279.SA2/GAP",
"XFEL.FEL/UNDULATOR.SASE2/BPS.2283.SA2/GAP",
"XFEL.FEL/UNDULATOR.SASE2/U40.2285.SA2/GAP",
"XFEL.FEL/UNDULATOR.SASE2/BPS.2289.SA2/GAP",
"XFEL.FEL/UNDULATOR.SASE2/U40.2291.SA2/GAP",
"XFEL.FEL/UNDULATOR.SASE2/BPS.2295.SA2/GAP",
"XFEL.FEL/UNDULATOR.SASE2/U40.2297.SA2/GAP",
"XFEL.FEL/UNDULATOR.SASE2/BPS.2307.SA2/GAP",
"XFEL.FEL/UNDULATOR.SASE2/U40.2310.SA2/GAP",
"XFEL.FEL/UNDULATOR.SASE2/BPS.2313.SA2/GAP",
"XFEL.FEL/UNDULATOR.SASE2/U40.2316.SA2/GAP",
"XFEL.FEL/UNDULATOR.SASE2/BPS.2319.SA2/GAP",
"XFEL.FEL/UNDULATOR.SASE2/U40.2322.SA2/GAP",
"XFEL.FEL/UNDULATOR.SASE2/BPS.2325.SA2/GAP",
"XFEL.FEL/UNDULATOR.SASE2/U40.2328.SA2/GAP",
"XFEL.FEL/UNDULATOR.SASE2/BPS.2331.SA2/GAP",
"XFEL.FEL/UNDULATOR.SASE2/U40.2334.SA2/GAP",
"XFEL.FEL/UNDULATOR.SASE2/BPS.2337.SA2/GAP",
"XFEL.FEL/UNDULATOR.SASE2/U40.2340.SA2/GAP",
"XFEL.FEL/UNDULATOR.SASE2/BPS.2344.SA2/GAP",
"XFEL.FEL/UNDULATOR.SASE2/U40.2346.SA2/GAP",
"XFEL.FEL/UNDULATOR.SASE2/BPS.2350.SA2/GAP",
"XFEL.FEL/UNDULATOR.SASE2/U40.2352.SA2/GAP",
"XFEL.FEL/UNDULATOR.SASE2/BPS.2356.SA2/GAP",
"XFEL.FEL/UNDULATOR.SASE2/U40.2358.SA2/GAP",
"XFEL.FEL/UNDULATOR.SASE2/BPS.2362.SA2/GAP",
"XFEL.FEL/UNDULATOR.SASE2/U40.2365.SA2/GAP",
"XFEL.FEL/UNDULATOR.SASE2/BPS.2368.SA2/GAP",
"XFEL.FEL/UNDULATOR.SASE2/U40.2371.SA2/GAP",
"XFEL.FEL/UNDULATOR.SASE2/BPS.2374.SA2/GAP",
"XFEL.FEL/UNDULATOR.SASE2/U40.2377.SA2/GAP",
"XFEL.FEL/UNDULATOR.SASE2/BPS.2380.SA2/GAP",
"XFEL.FEL/UNDULATOR.SASE2/U40.2383.SA2/GAP",
"XFEL.FEL/UNDULATOR.SASE2/BPS.2386.SA2/GAP",
"XFEL.FEL/UNDULATOR.SASE2/U40.2389.SA2/GAP",
"XFEL.FEL/UNDULATOR.SASE2/BPS.2392.SA2/GAP",
"XFEL.FEL/UNDULATOR.SASE2/U40.2395.SA2/GAP",
"XFEL.FEL/UNDULATOR.SASE2/BPS.2398.SA2/GAP",
"XFEL.FEL/UNDULATOR.SASE2/U40.2401.SA2/GAP",
"XFEL.FEL/UNDULATOR.SASE2/BPS.2404.SA2/GAP",
"XFEL.FEL/UNDULATOR.SASE2/U40.2407.SA2/GAP",
"XFEL.FEL/UNDULATOR.SASE2/BPS.2411.SA2/GAP",
"XFEL.FEL/UNDULATOR.SASE2/U40.2413.SA2/GAP",
"XFEL.FEL/UNDULATOR.SASE2/BPS.2417.SA2/GAP",
"XFEL.FEL/UNDULATOR.SASE2/U40.2419.SA2/GAP",
"XFEL.FEL/UNDULATOR.SASE3/U68.2809.SA3/GAP",
"XFEL.FEL/UNDULATOR.SASE3/BPS.2813.SA3/GAP",
"XFEL.FEL/UNDULATOR.SASE3/U68.2815.SA3/GAP",
"XFEL.FEL/UNDULATOR.SASE3/BPS.2819.SA3/GAP",
"XFEL.FEL/UNDULATOR.SASE3/U68.2821.SA3/GAP",
"XFEL.FEL/UNDULATOR.SASE3/BPS.2825.SA3/GAP",
"XFEL.FEL/UNDULATOR.SASE3/U68.2827.SA3/GAP",
"XFEL.FEL/UNDULATOR.SASE3/BPS.2831.SA3/GAP",
"XFEL.FEL/UNDULATOR.SASE3/U68.2834.SA3/GAP",
"XFEL.FEL/UNDULATOR.SASE3/BPS.2837.SA3/GAP",
"XFEL.FEL/UNDULATOR.SASE3/U68.2840.SA3/GAP",
"XFEL.FEL/UNDULATOR.SASE3/BPS.2843.SA3/GAP",
"XFEL.FEL/UNDULATOR.SASE3/U68.2846.SA3/GAP",
"XFEL.FEL/UNDULATOR.SASE3/BPS.2849.SA3/GAP",
"XFEL.FEL/UNDULATOR.SASE3/U68.2852.SA3/GAP",
"XFEL.FEL/UNDULATOR.SASE3/BPS.2855.SA3/GAP",
"XFEL.FEL/UNDULATOR.SASE3/U68.2858.SA3/GAP",
"XFEL.FEL/UNDULATOR.SASE3/BPS.2861.SA3/GAP",
"XFEL.FEL/UNDULATOR.SASE3/U68.2864.SA3/GAP",
"XFEL.FEL/UNDULATOR.SASE3/BPS.2867.SA3/GAP",
"XFEL.FEL/UNDULATOR.SASE3/U68.2870.SA3/GAP",
"XFEL.FEL/UNDULATOR.SASE3/BPS.2880.SA3/GAP",
"XFEL.FEL/UNDULATOR.SASE3/U68.2882.SA3/GAP",
"XFEL.FEL/UNDULATOR.SASE3/BPS.2886.SA3/GAP",
"XFEL.FEL/UNDULATOR.SASE3/U68.2888.SA3/GAP",
"XFEL.FEL/UNDULATOR.SASE3/BPS.2892.SA3/GAP",
"XFEL.FEL/UNDULATOR.SASE3/U68.2894.SA3/GAP",
"XFEL.FEL/UNDULATOR.SASE3/BPS.2898.SA3/GAP",
"XFEL.FEL/UNDULATOR.SASE3/U68.2901.SA3/GAP",
"XFEL.FEL/UNDULATOR.SASE3/BPS.2904.SA3/GAP",
"XFEL.FEL/UNDULATOR.SASE3/U68.2907.SA3/GAP",
"XFEL.FEL/UNDULATOR.SASE3/BPS.2910.SA3/GAP",
"XFEL.FEL/UNDULATOR.SASE3/U68.2913.SA3/GAP",
"XFEL.FEL/UNDULATOR.SASE3/BPS.2916.SA3/GAP",
"XFEL.FEL/UNDULATOR.SASE3/U68.2919.SA3/GAP",
"XFEL.FEL/UNDULATOR.SASE3/BPS.2922.SA3/GAP",
"XFEL.FEL/UNDULATOR.SASE3/U68.2925.SA3/GAP",
"XFEL.FEL/UNDULATOR.SASE3/BPS.2928.SA3/GAP",
"XFEL.FEL/UNDULATOR.SASE3/U68.2931.SA3/GAP",
"XFEL.FEL/UNDULATOR.SASE3/BPS.2934.SA3/GAP",
"XFEL.FEL/UNDULATOR.SASE3/U68.2937.SA3/GAP",
"XFEL.MAGNETS/CHICANE/HXRSS01/DT_FS",
"XFEL.MAGNETS/CHICANE/HXRSS02/DT_FS",
"XFEL.MAGNETS/CHICANE/SXR2CPP/DT_FS",
"XFEL.FEL/UNDULATOR.SASE2/MONOCC.2252.SA2/POS",
"XFEL.FEL/UNDULATOR.SASE2/MONOCI.2252.SA2/POS",
"XFEL.FEL/UNDULATOR.SASE2/MONOCC.2307.SA2/POS",
"XFEL.FEL/UNDULATOR.SASE2/MONOCI.2307.SA2/POS",
"XFEL.FEL/UNDULATOR.SASE2/MONORA.2252.SA2/ANGLE",
"XFEL.FEL/UNDULATOR.SASE2/MONOPA.2252.SA2/ANGLE",
"XFEL.FEL/UNDULATOR.SASE2/MONORA.2307.SA2/ANGLE",
"XFEL.FEL/UNDULATOR.SASE2/MONOPA.2307.SA2/ANGLE",
"XFEL.FEL/WAVELENGTHCONTROL.SA1/XFEL.SA1/E_PHOTON",
"XFEL.FEL/WAVELENGTHCONTROL.SA1/XFEL.SA1.COLOR1/E_PHOTON",
"XFEL.FEL/WAVELENGTHCONTROL.SA1/XFEL.SA1.COLOR2/E_PHOTON",
"XFEL.FEL/WAVELENGTHCONTROL.SA1/XFEL.SA1.COLOR3/E_PHOTON",
"XFEL.FEL/WAVELENGTHCONTROL.SA2/XFEL.SA2/E_PHOTON",
"XFEL.FEL/WAVELENGTHCONTROL.SA2/XFEL.SA2.COLOR1/E_PHOTON",
"XFEL.FEL/WAVELENGTHCONTROL.SA2/XFEL.SA2.COLOR2/E_PHOTON",
"XFEL.FEL/WAVELENGTHCONTROL.SA2/XFEL.SA2.COLOR3/E_PHOTON",
"XFEL.FEL/WAVELENGTHCONTROL.SA3/XFEL.SA3/E_PHOTON",
"XFEL.FEL/WAVELENGTHCONTROL.SA3/XFEL.SA3.COLOR1/E_PHOTON",
"XFEL.FEL/WAVELENGTHCONTROL.SA3/XFEL.SA3.COLOR2/E_PHOTON",
"XFEL.FEL/WAVELENGTHCONTROL.SA3/XFEL.SA3.COLOR3/E_PHOTON",
"XFEL.FEL/XGM/XGM.2643.T9/INTENSITY.SA1.SLOW.TRAIN",
"XFEL.FEL/XGM/XGM.2595.T6/INTENSITY.SLOW.TRAIN",
"XFEL.FEL/XGM/XGM.3130.T10/INTENSITY.SA3.SLOW.TRAIN",
"XFEL.FEEDBACK/FT2.LONGITUDINAL/MONITOR4/MEAN_AVG",
"XFEL.FEEDBACK/FT2.LONGITUDINAL/MONITOR6/MEAN_AVG",
"XFEL.FEEDBACK/FT2.LONGITUDINAL/MONITOR8/MEAN_AVG",
"XFEL.FEEDBACK/FT2.LONGITUDINAL/MONITOR11/MEAN_AVG",
"XFEL.FEEDBACK/FT2.LONGITUDINAL/MONITOR13/MEAN_AVG",
"XFEL.FEEDBACK/FT2.LONGITUDINAL/MONITOR15/MEAN_AVG",
"XFEL.FEEDBACK/FT2.LONGITUDINAL/MONITOR17/MEAN_AVG",
"XFEL.FEEDBACK/FT1.LONGITUDINAL/MONITOR4/MEAN_AVG",
"XFEL.FEEDBACK/FT1.LONGITUDINAL/MONITOR6/MEAN_AVG",
"XFEL.FEEDBACK/FT1.LONGITUDINAL/MONITOR8/MEAN_AVG",
"XFEL.FEEDBACK/FT1.LONGITUDINAL/MONITOR11/MEAN_AVG",
"XFEL.FEEDBACK/FT1.LONGITUDINAL/MONITOR13/MEAN_AVG",
"XFEL.FEEDBACK/FT1.LONGITUDINAL/MONITOR15/MEAN_AVG",
"XFEL.FEEDBACK/FT1.LONGITUDINAL/MONITOR17/MEAN_AVG"]

class XFELMachineInterface(MachineInterface):
    """
    Machine Interface for European XFEL
    """
    name = 'XFELMachineInterface'

    def __init__(self, args=None):
        super(XFELMachineInterface, self).__init__(args)
        if 'pydoocs' not in sys.modules:
            print('error importing doocs library')
        self.logbook_name = "xfellog"

        path2root = os.path.abspath(os.path.join(__file__ , "../../../.."))
        self.config_dir = os.path.join(path2root, "config_optim")

    def get_value(self, channel):
        """
        Getter function for XFEL.

        :param channel: (str) String of the devices name used in doocs
        :return: Data from pydoocs.read(), variable data type depending on channel
        """
        try:
            val = pydoocs.read(channel)
        except pydoocs.DoocsException:
            val={'data':float('nan')}
        return val["data"]

    def set_value(self, channel, val):
        """
        Method to set value to a channel

        :param channel: (str) String of the devices name used in doocs
        :param val: value
        :return: None
        """
        pydoocs.write(channel, val)
        return


    def get_charge(self):
        return self.get_value("XFEL.DIAG/CHARGE.ML/TORA.25.I1/CHARGE.SA1")

    def get_sases(self):
        try:
            sa1 = self.get_value("XFEL.FEL/XGM/XGM.2643.T9/INTENSITY.SA1.SLOW.TRAIN")
        except:
            sa1 = None
        try:
            sa2 = self.get_value("XFEL.FEL/XGM/XGM.2595.T6/INTENSITY.SLOW.TRAIN")
        except:
            sa2 = None
        try:
            sa3 = self.get_value("XFEL.FEL/XGM/XGM.3130.T10/INTENSITY.SA3.SLOW.TRAIN")
        except:
            sa3 = None
        return [sa1, sa2, sa3]

    def get_beam_energy(self):
        try:
            tld = self.get_value("XFEL.DIAG/BEAM_ENERGY_MEASUREMENT/TLD/ENERGY.DUD")
        except:
            tld = None
        #t3 = self.get_value("XFEL.DIAG/BEAM_ENERGY_MEASUREMENT/T3/ENERGY.SA2")
        #t4 = self.get_value("XFEL.DIAG/BEAM_ENERGY_MEASUREMENT/T4/ENERGY.SA1")
        #t5 = self.get_value("XFEL.DIAG/BEAM_ENERGY_MEASUREMENT/T5/ENERGY.SA2")
        try:
            t4d = self.get_value("XFEL.DIAG/BEAM_ENERGY_MEASUREMENT/T4D/ENERGY.SA1")
        except:
            t4d = None
        try:
            t5d = self.get_value("XFEL.DIAG/BEAM_ENERGY_MEASUREMENT/T5D/ENERGY.SA2")
        except:
            t5d = None
        return [tld, t4d, t5d]

    def get_wavelength(self):
        try:
            sa1 = self.get_value("XFEL.FEL/XGM.PHOTONFLUX/XGM.2643.T9/WAVELENGTH")
        except:
            sa1 = None
        try:
            sa2 = self.get_value("XFEL.FEL/XGM.PHOTONFLUX/XGM.2595.T6/WAVELENGTH")
        except:
            sa2 = None
        try:
            sa3 = self.get_value("XFEL.FEL/XGM.PHOTONFLUX/XGM.3130.T10/WAVELENGTH")
        except:
            sa3 = None
        return [sa1, sa2, sa3]

    def get_ref_sase_signal(self):
        try:
            sa1 = self.get_value("XFEL.FEL/XGM/XGM.2643.T9/INTENSITY.SA1.SLOW.TRAIN")
        except:
            sa1 = None
        try:
            sa2 = self.get_value("XFEL.FEL/XGM/XGM.2595.T6/INTENSITY.SLOW.TRAIN")
        except:
            sa2 = None
        #try:
        #    sa3 = self.get_value("XFEL.FEL/XGM.PHOTONFLUX/XGM.3130.T10/WAVELENGTH")
        #except:
        #    sa3 = None
        return [sa1, sa2]

    def write_data(self, method_name, objective_func, devices=[], maximization=False, max_iter=0):
        """
        Save optimization parameters to the Database

        :param method_name: (str) The used method name.
        :param objective_func: (Target) The Target class object.
        :param devices: (list) The list of devices on this run.
        :param maximization: (bool) Whether or not the data collection was a maximization. Default is False.
        :param max_iter: (int) Maximum number of Iterations. Default is 0.

        :return: status (bool), error_msg (str)
        """

        if objective_func is None:
            return False, "Objective Function required to save data."


        dump2json = {}

        for dev in devices:
            dump2json[dev.eid] = dev.values

        dump2json["method"] = method_name
        dump2json["dev_times"] = devices[0].times
        dump2json["obj_times"] = objective_func.times
        dump2json["maximization"] = maximization
        dump2json["nreadings"] = [objective_func.nreadings]
        dump2json["function"] = objective_func.eid
        dump2json["beam_energy"] = self.get_beam_energy()
        dump2json["wavelength"] = self.get_wavelength()
        dump2json["obj_values"] = np.array(objective_func.values).tolist()
        dump2json["std"] = np.array(objective_func.std_dev).tolist()
        try:
            dump2json["ref_sase"] = [objective_func.ref_sase[0], objective_func.ref_sase[-1]]
        except Exception as e:
            print("ERROR. Read ref sase: " + str(e))
            dump2json["ref_sase"] = [None]


        try:
            dump2json["charge"] = [self.get_charge()]
        except Exception as e:
            print("ERROR. Read charge: " + str(e))
            dump2json["charge"] = [None]

        if not os.path.exists(self.path2jsondir):
            os.makedirs(self.path2jsondir)

        filename = os.path.join(self.path2jsondir, datetime.now().strftime("%Y-%m-%d %H-%M-%S") + ".json")
        try:
            with open(filename, 'w') as f:
                json.dump(dump2json, f)
        except Exception as e:
            print("ERROR. Could not write data: " + str(e))
        return True, ""


    def send_to_logbook(self, *args, **kwargs):
        """
        UNUSED?
        Send information to the electronic logbook.

        :param args:
            Values sent to the method without keywork
        :param kwargs:
            Dictionary with key value pairs representing all the metadata
            that is available for the entry.
        :return: bool
            True when the entry was successfully generated, False otherwise.
        """
        author = kwargs.get('author', '')
        title = kwargs.get('title', '')
        severity = kwargs.get('severity', '')
        text = kwargs.get('text', '')
        image = kwargs.get('image', None)
        elog = self.logbook_name
        
        # The DOOCS elog expects an XML string in a particular format. This string
        # is beeing generated in the following as an initial list of strings.
        succeded = True  # indicator for a completely successful job
        # list beginning
        elogXMLStringList = ['<?xml version="1.0" encoding="ISO-8859-1"?>', '<entry>']

        # author information
        elogXMLStringList.append('<author>')
        elogXMLStringList.append(author)
        elogXMLStringList.append('</author>')
        # title information
        elogXMLStringList.append('<title>')
        elogXMLStringList.append(title)
        elogXMLStringList.append('</title>')
        # severity information
        elogXMLStringList.append('<severity>')
        elogXMLStringList.append(severity)
        elogXMLStringList.append('</severity>')
        # text information
        elogXMLStringList.append('<text>')
        elogXMLStringList.append(text)
        elogXMLStringList.append('</text>')
        # image information
        if image:
            try:
                encodedImage = base64.b64encode(image)
                elogXMLStringList.append('<image>')
                elogXMLStringList.append(encodedImage.decode())
                elogXMLStringList.append('</image>')
            except:  # make elog entry anyway, but return error (succeded = False)
                succeded = False
        # list end
        elogXMLStringList.append('</entry>')
        # join list to the final string
        elogXMLString = '\n'.join(elogXMLStringList)
        # open printer process
        try:
            lpr = subprocess.Popen(['/usr/bin/lp', '-o', 'raw', '-d', elog],
                                   stdin=subprocess.PIPE, stdout=subprocess.PIPE)
            # send printer job
            lpr.communicate(elogXMLString.encode('utf-8'))
        except:
            succeded = False
        return succeded

    def get_obj_function_module(self):
        from mint.xfel import xfel_obj_function
        return xfel_obj_function

    def get_preset_settings(self):
        """
        Return the preset settings to be assembled as Push Buttons at the user interface for quick load of settings.

        :return: (dict) Dictionary with Key being the group name and as value an array of dictionaries following the
        format:
            {"display": "Text of the PushButton", "filename": "my_file.json"}
        """
        presets = {
            "SASE1 opt 1": [
                {"display": "1. Launch orbit", "filename": "sase1_1.json"},
                {"display": "2. Match Quads", "filename": "sase1_2.json"},
                 {"display": "3. SASE1 CAX CAY", "filename": "SASE1_CAX_CAY.json"}],
        "SASE1 opt 2": [
                  {"display": "4. SASE1 CAX CBX", "filename": "SASE1_CAX_CBX.json"},
                {"display": "5. SASE1 phase-shifters", "filename": "SASE1_phase_shifter.json"},
                {"display": "6. SASE1 inj elements", "filename": "SASE1_tuning_with_injector_elements.json"},
            ],
            
            "SASE2 Opt": [
                 {"display": "1. Match Quads", "filename": "SASE2_matching_quads.json"},
                  {"display": "2. AirCoils", "filename": "SASE2_CAX_CBX_CAY_CBY.json"},
                  {"display": "3. Phase-shifters", "filename": "SASE2_BPS.json"},
            ],
            
            "Dispersion Minimization": [
                {"display": "1. I1 Horizontal", "filename": "disp_1.json"},
                {"display": "2. I1 Vertical", "filename": "disp_2.json"},
            ]
        }
        
        return presets

    def get_quick_add_devices(self):
        """
        Return a dictionary with:
        {
        "QUADS1" : ["...", "..."],
        "QUADS2": ["...", "..."]
        }

        That is converted into a combobox which allow users to easily populate the devices list

        :return: dict
        """


        devs = OrderedDict([
            ("Launch SASE1", ["XFEL.MAGNETS/MAGNET.ML/CFX.2162.T2/CURRENT.SP",
                               "XFEL.MAGNETS/MAGNET.ML/CFX.2219.T2/CURRENT.SP",
                               "XFEL.MAGNETS/MAGNET.ML/CFY.2177.T2/CURRENT.SP",
                               "XFEL.MAGNETS/MAGNET.ML/CFY.2207.T2/CURRENT.SP"]),

            ("Match Quads SASE1", ["XFEL.MAGNETS/MAGNET.ML/CFX.2162.T2/CURRENT.SP",
                                "XFEL.MAGNETS/MAGNET.ML/CFX.2219.T2/CURRENT.SP",
                                "XFEL.MAGNETS/MAGNET.ML/CFY.2177.T2/CURRENT.SP",
                                "XFEL.MAGNETS/MAGNET.ML/CFY.2207.T2/CURRENT.SP"]),
            ("I1 Hor. Disp.", ["XFEL.MAGNETS/MAGNET.ML/CBB.62.I1D/KICK_MRAD.SP",
                                "XFEL.MAGNETS/MAGNET.ML/CIX.90.I1/KICK_MRAD.SP",
                                "XFEL.MAGNETS/MAGNET.ML/CIX.95.I1/KICK_MRAD.SP",
                                "XFEL.MAGNETS/MAGNET.ML/CIX.65.I1/KICK_MRAD.SP",
                                "XFEL.MAGNETS/MAGNET.ML/CIX.51.I1/KICK_MRAD.SP",
                                "XFEL.MAGNETS/MAGNET.ML/CIX.102.I1/KICK_MRAD.SP",
                                "XFEL.MAGNETS/MAGNET.ML/CX.39.I1/KICK_MRAD.SP",
                                "XFEL.MAGNETS/MAGNET.ML/BL.50I.I1/KICK_DEG.SP",
                                "XFEL.MAGNETS/MAGNET.ML/BL.50II.I1/KICK_DEG.SP"]),
            ("I1 Ver. Disp.", ["XFEL.MAGNETS/MAGNET.ML/CIY.92.I1/KICK_MRAD.SP",
                                "XFEL.MAGNETS/MAGNET.ML/CIY.72.I1/KICK_MRAD.SP",
                                "XFEL.MAGNETS/MAGNET.ML/CY.39.I1/KICK_MRAD.SP"])
        ])
        return None
# test interface


class TestMachineInterface(XFELMachineInterface):
    """
    Machine interface for testing
    """
    name = 'TestMachineInterface'

    def __init__(self, args):
        super(TestMachineInterface, self).__init__(args)
        self.data = 1.
        pass

    def get_alarms(self):
        return np.random.rand(4)#0.0, 0.0, 0.0, 0.0]

    def get_value(self, device_name):
        """
        Testing getter function for XFEL.

        :param channel: (str) String of the devices name used in doocs
        :return: Data from pydoocs.read(), variable data type depending on channel
        """
        #if "QUAD" in device_name:
        #    return 0
        #spectrum  = 10*np.exp(-np.linspace(-10, 10, num=1280)**2/((2*2))) + 10*np.exp(-np.linspace(-8, 12, num=1280)**2/((2*0.25))) + np.random.rand(1280)
        val = np.random.random()
        return  val

    def set_value(self, device_name, val):
        """
        Testing Method to set value to a channel

        :param channel: (str) String of the devices name used in doocs
        :param val: value
        :return: None
        """
        #print("set:", device_name,  "-->", val)
        self.data += np.sqrt(val**2)
        return 0.0

    def get_bpms_xy(self, bpms):
        """
        Testing method for getting bmps data

        :param bpms: list of string. BPMs names
        :return: X, Y - two arrays in [m]
        """
        X = np.zeros(len(bpms))
        Y = np.zeros(len(bpms))
        return X, Y


    @staticmethod
    def send_to_logbook(*args, **kwargs):
        """
        Send information to the electronic logbook.

        :param args:
            Values sent to the method without keywork
        :param kwargs:
            Dictionary with key value pairs representing all the metadata
            that is available for the entry.
        :return: bool
            True when the entry was successfully generated, False otherwise.
        """
        author = kwargs.get('author', '')
        title = kwargs.get('title', '')
        severity = kwargs.get('severity', '')
        text = kwargs.get('text', '')
        elog = kwargs.get('elog', '')
        image = kwargs.get('image', None)

        print('Send to Logbook not implemented for TestMachineInterface.')
        return True

    def get_obj_function_module(self):
        from mint.xfel import xfel_obj_function
        return xfel_obj_function

    def device_factory(self, pv):
        """
        Create a device for the given PV using the proper Device Class.

        :param pv: (str) The process variable for which to create the device.
        :return: (Device) The device instance for the given PV.
        """
        return TestDevice(eid=pv)
