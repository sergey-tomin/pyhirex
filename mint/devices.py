"""
Sergey Tomin, XFEL/DESY, 2017
"""
from mint.opt_objects import Device
from PyQt5 import QtGui, QtCore
import numpy as np
import time
from threading import Thread, Event
from scipy.optimize import curve_fit
import logging
from mint.sase_imitation import imitate_1d_sase

logger = logging.getLogger(__name__)

class Spectrometer():
    def __init__(self, mi, eid=None, **kwargs):
        self.eid = eid
        self.mi = mi
        self.devmode = False
        self.num_px = 1280  # number of pixels
        self.x_axis = np.arange(self.num_px)
        self.spectrum = []
        self.background = []
        self.av_spectrum = []
        self.gauss_coeff_fit = None
        #self.update_params(transmission=1, calib_energy_coef=1)
        self.update_background()

    def is_online(self):
        if self.eid is not None and self.eid != "":
            try:
                self.mi.get_value(self.eid)
                status = True
            except:
                status = False
        else:
            status = False
        return status

    def update_params(self, transmission=1, calib_energy_coef=1):
        self.transmission = transmission
        self.calib_energy_coef = calib_energy_coef

    def update_background(self, background=None):
        if background is not None and len(background) == self.num_px:
            self.background = background
        else:
            self.background = np.zeros(self.num_px)

    def get_value(self):
        """
        basic method to get value/spectrum via DOOCS server
        :return:
        """
        if self.devmode:
            spectrum = 10 * np.exp(-np.linspace(-10, 10, num=1280) ** 2 / ((2 * 2))) + 10 * np.exp(
                -np.linspace(-8, 12, num=1280) ** 2 / ((2 * 0.25))) + np.random.rand(1280)
            return spectrum

        val = self.mi.get_value(self.eid)
        return val

    def get_spectrum(self):
        """
        basic method to get value/spectrum via DOOCS server
        :return:
        """
        raw_spectrum = self.get_value()

        if len(raw_spectrum) == len(self.background):
            spectrum = raw_spectrum - self.background
        else:
            spectrum = raw_spectrum
        spectrum = spectrum * self.calib_energy_coef * self.transmission
        return spectrum

    def cross_calibrate(self, spectrum, x_axis, transmission, pulse_energy):
        """
        Cross calibrate with spectrum with pulse energy

        :param spectrum: array
        :param transmission: transmission coefficient 0 - 1
        :param pulse_energy: in [uJ]
        :return: calibration coefficient
        """
        if len(spectrum) < 3:
            return
        #energy_uJ = self.mi.get_value(self.slow_xgm_signal)
        ave_integ = np.trapz(spectrum, x_axis)/transmission
        self.calib_energy_coef = pulse_energy/ave_integ
        return self.calib_energy_coef # uJ/au

    def fit_guass(self, spectrum):
        """
        method to fit gaus to average spectrum to find a central pixel

        :param spectrum: array
        :return: number of central pixel
        """
        if len(spectrum) == 0:
            return
        y = spectrum
        x = np.arange(len(y))

        def gauss(x, *p):
            A, mu, sigma = p
            return A * np.exp(-(x - mu) ** 2 / (2. * sigma ** 2))

        # (A, mu, sigma)
        p0 = [np.max(y), np.argmax(y), 30]

        self.gauss_coeff_fit, var_matrix = curve_fit(gauss, x, y, p0=p0)
        mu = self.gauss_coeff_fit[1]
        px1 = mu
        return px1

    def calibrate_axis(self, ev_px=None, E0=None, px1=None):
        """
        method to calibrate energy axis

        :param ev_px: ev/pixel
        :param E0: ev, energy of a central pixel
        :param px1: int, number of a centrl pixel
        :return: x_axis - array in [ev]
        """
        #ev_px = self.ui.sb_ev_px.value()
        #E0 = self.ui.sb_E0.value()
        #px1 = self.ui.sb_px1.value()
        start = E0 - px1*ev_px
        stop = E0 + (self.num_px - px1) * ev_px
        self.x_axis = np.linspace(start, stop, num=self.num_px)
        return self.x_axis


class BraggCamera(Spectrometer):
    def __init__(self, mi, eid=None, energy_ch=None):
        super(BraggCamera, self).__init__(mi=mi, eid=eid)
        self.mi = mi
        self.energy_ch = energy_ch
        self.eid = eid
        self.num_px = 1079  # number of pixels
        self.x_axis = np.arange(self.num_px)
        self.spectrum = []
        self.background = []
        self.av_spectrum = []
        self.update_background()

    def calibrate_axis(self):
        """
        method to calibrate energy axis

        :return: x_axis - array in [ev]
        """

        self.x_axis = self.mi.get_value(self.energy_ch)
        return self.x_axis


class SpectrometerSA3(Spectrometer):
    def __init__(self, mi, eid=None, energy_ch=None, **kwargs):
        super(SpectrometerSA3, self).__init__(mi=mi, eid=eid, **kwargs)
        self.mi = mi
        self.energy_ch = energy_ch
        self.eid = eid
        self.num_px = 1079  # number of pixels
        self.x_axis = np.arange(self.num_px)
        self.spectrum = []
        self.background = []
        self.av_spectrum = []
        self.update_background()


    def is_online(self):
        if self.energy_ch is not None and self.energy_ch != "":
            try:
                self.mi.get_value(self.energy_ch)
                status = True
            except:
                status = False
        else:
            status = False
        return status

    def calibrate_axis(self, ev_px=None, E0=None, px1=None):
        """
        method to calibrate energy axis

        :return: x_axis - array in [ev]
        """

        self.x_axis = self.mi.get_value(self.energy_ch)
        return self.x_axis

class SpectrometerViking(Spectrometer):
    def __init__(self, mi, eid=None, energy_ch=None, **kwargs):
        super(SpectrometerViking, self).__init__(mi=mi, eid=eid, **kwargs)
        self.mi = mi
        self.energy_ch = energy_ch
        self.eid = eid
        self.num_px = 2048  # number of pixels
        self.x_axis = np.arange(self.num_px)
        self.spectrum = []
        self.background = []
        self.av_spectrum = []
        self.update_background()


    def is_online(self):
        if self.energy_ch is not None and self.energy_ch != "":
            try:
                self.mi.get_value(self.energy_ch)
                status = True
            except:
                status = False
        else:
            status = False
        return status

    def calibrate_axis(self, ev_px=None, E0=None, px1=None):
        """
        method to calibrate energy axis

        :return: x_axis - array in [ev]
        """

        self.x_axis = self.mi.get_value(self.energy_ch)
        return self.x_axis

class CrazySpectrometer(Spectrometer):
    def __init__(self, mi, eid=None, energy_ch=None, **kwargs):
        super(CrazySpectrometer, self).__init__(mi=mi, eid=eid, **kwargs)
        #self.devmode = True
        self.mi = mi
        self.energy_ch = energy_ch
        self.eid = eid
        self.num_px = len(self.get_value())  # number of pixels
        self.x_axis = np.arange(self.num_px)
        self.spectrum = []
        self.background = []
        self.av_spectrum = []
        self.update_background()


    def is_online(self):
        if self.eid is not None and self.eid != "":
            try:
                self.mi.get_value(self.eid)
                status = True
            except:
                status = False
        else:
            status = False
        return status

    def calibrate_axis(self, ev_px=None, E0=None, px1=None):
        """
        method to calibrate energy axis

        :return: x_axis - array in [ev]
        """
        if self.energy_ch is None:
            val = self.get_value()
            self.x_axis = np.arange(len(val))
        else:
            self.x_axis = self.mi.get_value(self.energy_ch)
        return self.x_axis



class DummyHirex(Spectrometer):

    def __init__(self, *args, **kwargs):
        super(DummyHirex, self).__init__(*args, **kwargs)

    def actuator(self):
        return np.sin(time.time()/10)*2

    def get_value(self):
        """
        basic method to get value/spectrum via DOOCS server
        :return:
        """

        spectrum_phen = np.linspace(8800, 9200, 1280)

        sase_center = 9000
        sase_sigma = 20

        seed_center = 9020 + 10 * self.actuator()
        seed_power = np.exp(-(seed_center - sase_center - 10)**2 / (2 * sase_sigma/2)**2) * np.abs(np.random.randn(1)[0])
        seed_sigma = 1

        spectrum_sase = np.exp(-(spectrum_phen - sase_center)**2 / (2 * sase_sigma)**2)
        spectrum_seed = np.exp(-(spectrum_phen - seed_center)**2 / (2 * seed_sigma)**2)
        spectrum_noise = np.random.rand(len(spectrum_phen))

        val =  3 * spectrum_sase + 20 * seed_power * spectrum_seed + spectrum_noise * 3
        return val

    def is_online(self):
        return True


class DummySASE(Spectrometer):

    def __init__(self, *args, **kwargs):
        super(DummySASE, self).__init__(*args, **kwargs)

        self.sase_center = 9000
        self.sase_sigma = 10

        self.n_events = 1000

        _, _, spectrum_phen, fd = imitate_1d_sase(spec_center=self.sase_center, spec_res=0.1, spec_width=self.sase_sigma, spec_range=(8950, 9050), pulse_length=0.3*5, # 1 fs = 0.3 um
                    en_pulse=10e-6, flattop=0, n_events=self.n_events)

        self.idx=0

        self.spectrum_sase = abs(fd)**2
        self.spectrum_phen = spectrum_phen

        self.num_px = len(self.spectrum_phen)

        # spectrum_sase = np.exp(-(spectrum_phen - sase_center)**2 / (2 * sase_sigma)**2)
        # spectrum_seed = np.exp(-(spectrum_phen - seed_center)**2 / (2 * seed_sigma)**2)
        # spectrum_noise = np.random.rand(len(spectrum_phen))

        # val =  spectrum_sase + 2 * seed_power * spectrum_seed + spectrum_noise * 3

    def actuator(self):
        return np.sin(time.time()/10)*2

    def get_value(self):
        """
        basic method to get value/spectrum via DOOCS server
        :return:
        """

        # spectrum_phen = np.linspace(8800, 9200, 1280)
        #self.px_last = self.px_last if self.px_last != 0 else None
        spectrum_sase = self.spectrum_sase[:,self.idx % self.n_events]
        self.idx += 1

        seed_center = 9000 + 10 * self.actuator()
        # seed_power = np.exp(-(seed_center - self.sase_center - 10)**2 / (2 * self.sase_sigma/2)**2) * np.abs(np.random.randn(1)[0])
        seed_sigma = 1
        seed_power=1

        # spectrum_sase = np.exp(-(spectrum_phen - sase_center)**2 / (2 * sase_sigma)**2)
        spectrum_seed = np.exp(-(self.spectrum_phen - seed_center)**2 / (2 * seed_sigma)**2)
        spectrum_seed = spectrum_seed / np.amax(spectrum_seed) * 2
        # spectrum_noise = np.random.rand(len(self.spectrum_phen))

        # val =  spectrum_sase + 2 * seed_power * spectrum_seed + spectrum_noise * 3
        val = spectrum_sase# * spectrum_seed
        return val#[self.px_first:self.px_last]
        
        
    def is_online(self):
        return True
        
    def calibrate_axis(self, ev_px=None, E0=None, px1=None):
        self.x_axis = self.spectrum_phen
        return self.x_axis



class XGM():
    def __init__(self, mi, eid):
        self.mi = mi
        self.eid = eid

    def get_value(self):
        """
        basic method to get value from XFGM
        :return: val
        """
        try:
            val = self.mi.get_value(self.eid)
        except:
            val = np.nan
        return val


class DummyXGM(XGM):
    def __init__(self, *args, **kwargs):
        super(DummyXGM, self).__init__(*args, **kwargs)

    def get_value(self):
        """
        basic method to get value from XFGM
        :return: val
        """
        return 1000.


class BunchNumberCTRL():
    def __init__(self, mi, doocs_ch):
        self.mi = mi
        self.doocs_ch = doocs_ch

    def get_value(self):
        if self.doocs_ch is None:
            return 1
        val = self.mi.get_value(self.doocs_ch)
        return val

    def set_value(self, num):
        if self.doocs_ch is None:
            return
        self.mi.set_value(self.doocs_ch, num)


class Corrector(Device):
    def __init__(self, eid=None, server="XFEL", subtrain="SA1"):
        super(Corrector, self).__init__(eid=eid)
        self.subtrain = subtrain
        self.server = server

    def set_value(self, val):
        #self.values.append(val)
        #self.times.append(time.time())
        ch = self.server + ".MAGNETS/MAGNET.ML/" + self.eid + "/KICK_MRAD.SP"
        self.mi.set_value(ch, val)

    def get_value(self):
        ch = self.server + ".MAGNETS/MAGNET.ML/" + self.eid + "/KICK_MRAD.SP"
        val = self.mi.get_value(ch)
        return val

    def get_limits(self):
        ch_min = self.server+ ".MAGNETS/MAGNET.ML/" + self.id + "/MIN_KICK"
        min_kick = self.mi.get_value(ch_min)
        ch_max = self.server + ".MAGNETS/MAGNET.ML/" + self.id + "/MAX_KICK"
        max_kick = self.mi.get_value(ch_max)
        return [min_kick*1000, max_kick*1000]

    def is_ok(self):
        ch = self.server+ ".MAGNETS/MAGNET.ML/" + self.id + "/COMBINED_STATUS"
        status = int(self.mi.get_value(ch))
        power_bit = '{0:08b}'.format(status)[-2]
        busy_bit = '{0:08b}'.format(status)[-4]

        if power_bit == "1" and busy_bit == "0":
            return True
        else:
            return False



class ChargeDoocs(Device):
    def __init__(self, eid="XFEL.FEEDBACK/FT1.LONGITUDINAL/MONITOR1/TARGET", server="XFEL", subtrain="SA1"):
        super(ChargeDoocs, self).__init__(eid=eid)


class MPS(Device):
    def __init__(self, eid=None, server="XFEL", subtrain="SA1"):
        super(MPS, self).__init__(eid=eid)
        self.subtrain = subtrain
        self.server = server

    def beam_off(self):
        self.mi.set_value(self.server + ".UTIL/BUNCH_PATTERN/CONTROL/BEAM_ALLOWED", 0)

    def beam_on(self):
        self.mi.set_value(self.server + ".UTIL/BUNCH_PATTERN/CONTROL/BEAM_ALLOWED", 1)

    def num_bunches_requested(self, num_bunches=1):
        self.mi.set_value(self.server + ".UTIL/BUNCH_PATTERN/CONTROL/NUM_BUNCHES_REQUESTED_1", num_bunches)

    def is_beam_on(self):
        val = self.mi.get_value(self.server + ".UTIL/BUNCH_PATTERN/CONTROL/BEAM_ALLOWED")
        return val




class MIStandardFeedback(Device):
    def __init__(self, eid=None, server="XFEL", subtrain="SA1"):
        super(MIStandardFeedback, self).__init__(eid=eid)
        self.subtrain = subtrain
        self.server = server

    def is_running(self):
        status = self.mi.get_value(self.server + ".FEEDBACK/ORBIT.SA1/ORBITFEEDBACK/ACTIVATE_FB")
        return status


class MISASE3Feedback(Device):
    def __init__(self, eid=None, server="XFEL", subtrain="SA1"):
        super(MISASE3Feedback, self).__init__(eid=eid)
        self.subtrain = subtrain
        self.server = server

    def is_running(self):
        status = self.mi.get_value(self.server + ".FEEDBACK/ORBIT.SA3/ORBITFEEDBACK/ACTIVATE_FB")
        return status


class MISASE2Feedback(Device):
    def __init__(self, eid=None, server="XFEL", subtrain="SA1"):
        super(MISASE2Feedback, self).__init__(eid=eid)
        self.subtrain = subtrain
        self.server = server

    def is_running(self):
        status = self.mi.get_value(self.server + ".FEEDBACK/ORBIT.SA2/ORBITFEEDBACK/ACTIVATE_FB")
        return status
