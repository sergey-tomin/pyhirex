#!/opt/anaconda4/bin/python
"""
Created on Sun Aug  9 17:33:30 2020

@author: Sergey Tomin
"""
from PyQt5.QtWidgets import QFrame, QMainWindow
import sys
import os
import argparse
import time
try:
    old_scipy = False
    from scipy.signal import find_peaks
except:
    old_scipy = True

import numpy as np
import pyqtgraph as pg
from scipy.optimize import curve_fit
from threading import Thread, Event
path = os.path.realpath(__file__)
indx = path.find("hirex.py")
#print("PATH to main file: " + os.path.realpath(__file__) + " path to folder: "+ path[:indx])
sys.path.insert(0, path[:indx])
from matplotlib import cm
from gui.spectr_gui import *
from mint.xfel_interface import *
from gui.settings_gui import *
from mint.devices import Spectrometer, BunchNumberCTRL, DummyHirex, XGM, DummyXGM, DummySASE, SpectrometerViking, SpectrometerSA3, CrazySpectrometer
from scan import ScanInterface
from correlation import CorrelInterface
from correlation_2d import Correl2DInterface
from analysis_spec import AnalysisInterface
from logger import UILogger
from calculator import UICalculator
from scipy import ndimage
import pathlib
import json
import logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)
from opt_lib import fwhm3

AVAILABLE_MACHINE_INTERFACES = [XFELMachineInterface, TestMachineInterface]
AVAILABLE_SPECTROMETERS = ["SASE1", "SASE2", "SASE3", "SASE3_SCS", "VIKING", "DUMMY", "DUMMYSASE", "TEST1", "TEST2", "TEST3"]
#HIREX_N_PIXELS = 1280
#DOOCS_CTRL_N_BUNCH = "XFEL.UTIL/BUNCH_PATTERN/CONTROL/NUM_BUNCHES_REQUESTED_2"
DIR_NAME = "hirex"

PY_SPECTROMETER_DIR = "pySpectrometer"


class Background(Thread):
    def __init__(self, mi, device, dev_name):
        super(Background, self).__init__()
        self.mi = mi
        self.devmode = False
        self.device = device
        self.nshots = 100
        self.background = []
        self._stop_event = Event()
        self.dev_name = dev_name

    def load(self):
        self.background = np.array([])
        try:
            self.background = np.loadtxt(self.dev_name + "_background.txt")
        except Exception as ex:
            print("Problem with background: {}. Exception was: {}".format(self.dev_name + "_background.txt", ex))

        return self.background

    def run(self):
        Y = []
        for i in range(self.nshots + 1):
            x = self.device.get_value()
            # if self.devmode:
                # x = np.zeros_like(x) + 3 * np.exp(-np.linspace(-10, 10, num=len(x)) ** 2 / ((1 * 2)))
            # if i == 0:
                # continue
            Y.append(x)
            time.sleep(0.1)
        self.background = np.mean(Y, axis=0)
        np.savetxt(self.dev_name + "_background.txt", self.background)
        #time.sleep(0.5)
        print("Background finished")

    def stop(self):
        print("stop")
        self._stop_event.set()


class Transmission(Thread):
    def __init__(self, mi, dev_ch):
        super(Transmission, self).__init__()
        self.mi = mi
        self.devmode = False
        self._stop_event = Event()
        self.dev_ch = dev_ch
        self.transmission = 1.
        self.kill = False

    def run(self):
        while not self.kill:
            if self.dev_ch is not None and self.dev_ch != "":
                self.transmission = self.mi.get_value(self.dev_ch)

            time.sleep(2)

    def stop(self):
        print("stop transmission thread")
        self._stop_event.set()


class EnergyAxisWatcher(Thread):
    def __init__(self, mi, dev_ch):
        super(EnergyAxisWatcher, self).__init__()
        self.mi = mi
        self.devmode = False
        self._stop_event = Event()
        self.dev_ch = dev_ch
        self.energy_axis = []
        self.energy_axis_old = None
        self.trigger = False
        self.kill = False

    def is_online(self):
        if self.dev_ch is not None and self.dev_ch != "":
            try:
                self.mi.get_value(self.dev_ch)
                status = True
            except:
                status = False
        else:
            status = False
        return status


    def run(self):
        while not self.kill:
            if self.dev_ch is not None and self.dev_ch != "":
                self.energy_axis = self.mi.get_value(self.dev_ch)
                if self.energy_axis_old is None:
                    self.energy_axis_old = self.energy_axis
            else:
                break
            time.sleep(1)

            d_ev = (self.energy_axis[1] - self.energy_axis[0]) * 2 #TODO: check for stbility. 2 pixels scale noise
            if self.energy_axis[0] - d_ev/2. <= self.energy_axis_old[0] <= self.energy_axis[0] + d_ev/2:
                self.trigger = False
            else:
                self.trigger = True
                print("Photon energy changed from", self.energy_axis[0], 'by', self.energy_axis_old[1]-self.energy_axis[0])
            self.energy_axis_old = self.energy_axis

    def stop(self):
        print("stop EnergyAxisWatcher thread")
        self._stop_event.set()



class SpectrometerWindow(QMainWindow):
    """ Main class for the GUI application """
    def __init__(self):
        """
        Initialize the GUI and QT UI aspects of the application.
        Initialize the scan parameters.
        Connect start and logbook buttons on the scan panel.
        Initialize the plotting.
        Make the timer object that updates GUI on clock cycle during a scan.
        """
        # PATHS

        self.tool_args = None
        self.parse_arguments()
        self.dev_mode = self.tool_args.devmode

        args = vars(self.tool_args)
        if self.dev_mode:
            self.mi = TestMachineInterface(args)
        else:
            class_name = self.tool_args.mi
            #print(class_name)
            if class_name not in globals():
                print("Could not find Machine Interface with name: {}. Loading XFELMachineInterface instead.".format(class_name))
                self.mi = XFELMachineInterface(args)
            else:
                self.mi = globals()[class_name](args)
        DIR_NAME = os.path.basename(pathlib.Path(__file__).parent.absolute())
        print("PATH" , DIR_NAME)
        self.path = path[:path.find(DIR_NAME)]
        self.config_dir = self.path + DIR_NAME + os.sep + "configs" + os.sep
        self.settings_file = self.config_dir + "settings.json"
        self.gui_dir = self.path + DIR_NAME + os.sep + "gui" + os.sep
        self.gui_styles = ["standard.css", "colinDark.css", "dark.css"]
        #self.data_dir = self.path + DIR_NAME + os.sep + "configs" + os.sep
        # initialize
        QFrame.__init__(self)
        self.ui = MainWindow(self)

        self.settings = None
        #self.load_settings()
        is_spectrometer = False
        for hirex in AVAILABLE_SPECTROMETERS:
            if self.tool_args.__dict__[hirex]:
                is_spectrometer = True
                self.ui.combo_hirex.addItem(hirex)
        if not is_spectrometer:
            # self.ui.combo_hirex.addItem("DUMMY")
            self.ui.combo_hirex.addItem("DUMMYSASE")
        #self.ui.combo_hirex.addItem("SASE2 HIREX")
        #self.ui.combo_hirex.addItem("SASE1 HIREX")
        current_source = self.ui.combo_hirex.currentText()
        self.config_file = self.config_dir + current_source + "_config.json"

        self.data_dir = path[:path.find("user")]  + "user" + os.sep + PY_SPECTROMETER_DIR + os.sep + current_source + os.sep
        self.px_first = 0
        self.px_last = None
        self.ui.combo_hirex.currentIndexChanged.connect(self.reload_objects_settings)
        self.reload_objects_settings()

        self.scantool = ScanInterface(parent=self)
        self.corretool = CorrelInterface(parent=self)
        self.corre2dtool = Correl2DInterface(parent=self)
        self.analysistool = AnalysisInterface(parent=self)


        self.add_plot()
        self.add_image_widget()
        self.ui.restore_state(self.config_file)

        self.data_2d = np.zeros((self.spectrometer.num_px, int(self.sb_2d_hist_size)))

        self.timer_live = pg.QtCore.QTimer()
        self.timer_live.timeout.connect(self.get_transmission)
        self.timer_live.timeout.connect(self.calc_spec)
        self.timer_plot = pg.QtCore.QTimer()
        self.timer_plot.timeout.connect(self.plot_spec)


        self.ui.pb_start.clicked.connect(self.start_stop_live_spectrum)
        self.ui.pb_background.clicked.connect(self.take_background)
        self.spectrum_list = []
        self.ave_spectrum = []
        self.peak_ev = None
        self.spectrum_event = None
        self.spectrum_event_disp = None
        self.peak_ev = 0 #position of peak in eV (middle of fwhm)
        self.fwhm_ev = 0 # fwhm width in eV ()
        self.ave_integ = 0 # pulse energy from spectrometer integral
        self.pulse_energy = 0 # pulse energy from XGM
        self.counter_spect = 0

        self.background = self.back_taker.load()

        self.ui.sb_ev_px.valueChanged.connect(self.calibrate_axis)
        self.ui.sb_E0.valueChanged.connect(self.calibrate_axis)
        self.ui.sb_px1.valueChanged.connect(self.calibrate_axis)

        #self.spectrometer.calibrate_axis(ev_px=self.ui.sb_ev_px.value(), E0=self.ui.sb_E0.value(), px1=self.ui.sb_px1.value())
        self.calibrate_axis()
        self.gauss_coeff_fit = None
        self.ui.pb_estim_px1.clicked.connect(self.fit_guass)
        self.ui.chb_show_fit.stateChanged.connect(self.show_fit)
        self.ui.chb_uj_ev.stateChanged.connect(self.set_labels)
        self.ui.chb_uj_ev.stateChanged.connect(self.analysistool.reset_spectra)

        # self.ui.chb_uj_ev.stateChanged.connect(self.show_fit) # ##################################################################################################################################

        self.back_taker_status = pg.QtCore.QTimer()
        self.back_taker_status.timeout.connect(self.is_back_taker_alive)

        self.ui.actionSettings.triggered.connect(self.run_settings_window)
        self.calculator_window = None
        self.ui.actionSelf_Seeding_tools.triggered.connect(self.run_calculator_window)
        self.ui.pb_cross_calib.clicked.connect(self.cross_calibrate)
        self.calib_energy_coef = 1
        self.plot1.scene().sigMouseMoved.connect(self.mouseMoved)
        #proxy = pg.SignalProxy(self.plot1.scene().sigMouseMoved, rateLimit=60, slot=self.mouseMoved)

        self.ui.actionSave_Data.triggered.connect(self.save_data_as)
        self.ui.pb_hide_show_backplot.clicked.connect(self.show_hide_background)
        self.ui.pb_hide_average.clicked.connect(self.show_hide_average)
        self.check_doocs_permission()
        self.ui.sb_px_first.valueChanged.connect(self.reset_waterfall)
        self.ui.sb_px_last.valueChanged.connect(self.reset_waterfall)
        self.logger_window = None
        self.ui.pb_logger.clicked.connect(self.run_logger_window)


    def check_doocs_permission(self):
        self.doocs_permit = True
        try:
            self.mi.set_value("XFEL.UTIL/DYNPROP/MISC/HIREX_PY", 0.0)
        except:
            self.doocs_permit = False
        if not self.doocs_permit:
            self.ui.groupBox_5.setTitle("Control: no permission to write to DOOCS")
            self.ui.groupBox_5.setStyleSheet('QGroupBox  {color: red;}')


    def run_logger_window(self):
        if self.logger_window is None:
            self.logger_window = UILogger(parent=self)
        self.logger_window.show()

    def reload_objects_settings(self):
        try:
            self.load_settings()
        except:
            self.run_settings_window()
            self.settings.apply_settings()
        self.load_objects()



    def load_objects(self):

        current_source = self.ui.combo_hirex.currentText()
        print(current_source)
        if current_source in ["SASE2", "SASE1"]:

            self.bunch_num_ctrl = BunchNumberCTRL(self.mi, self.doocs_ctrl_num_bunch)

            self.spectrometer = Spectrometer(self.mi, eid=self.hirex_doocs_ch)
            self.spectrometer.num_px = self.hrx_n_px
            self.spectrometer.devmode = self.dev_mode
            self.xgm = XGM(mi=self.mi, eid=self.slow_xgm_signal)

        elif current_source in ["SASE3","SASE3_SCS"]:

            self.bunch_num_ctrl = BunchNumberCTRL(self.mi, self.doocs_ctrl_num_bunch)

            self.spectrometer = SpectrometerSA3(self.mi, energy_ch=self.ph_energy_ch, eid=self.hirex_doocs_ch)
            self.spectrometer.num_px = self.hrx_n_px
            self.spectrometer.devmode = self.dev_mode
            self.xgm = XGM(mi=self.mi, eid=self.slow_xgm_signal)

        elif current_source in ["VIKING"]:
            self.bunch_num_ctrl = BunchNumberCTRL(self.mi, self.doocs_ctrl_num_bunch)
            self.spectrometer = SpectrometerViking(self.mi, energy_ch=self.ph_energy_ch, eid=self.hirex_doocs_ch)
            self.spectrometer.num_px = 2048
            self.spectrometer.devmode = self.dev_mode
            self.xgm = XGM(mi=self.mi, eid=self.slow_xgm_signal)

        elif current_source in ["DUMMY"]:
            self.bunch_num_ctrl = BunchNumberCTRL(self.mi, None) # delete

            self.spectrometer = DummyHirex(self.mi, eid=self.hirex_doocs_ch)
            self.xgm = DummyXGM(mi=self.mi, eid=self.slow_xgm_signal)

        elif current_source in ["DUMMYSASE"]:
            self.bunch_num_ctrl = BunchNumberCTRL(self.mi, None) # delete
            self.spectrometer = DummySASE(self.mi, eid=self.hirex_doocs_ch)
            self.xgm = DummyXGM(mi=self.mi, eid=self.slow_xgm_signal)
        elif current_source in ["TEST1", "TEST2", "TEST3"]:
            self.bunch_num_ctrl = BunchNumberCTRL(self.mi, self.doocs_ctrl_num_bunch)

            self.spectrometer = CrazySpectrometer(self.mi, energy_ch=self.ph_energy_ch, eid=self.hirex_doocs_ch)
            self.spectrometer.devmode = self.dev_mode
            self.xgm = XGM(mi=self.mi, eid=self.slow_xgm_signal)

        self.back_taker = Background(mi=self.mi, device=self.spectrometer,  dev_name=current_source)
        self.background = self.back_taker.load()

        self.config_file = self.config_dir + current_source + "_config.json"
        self.ui.restore_state(self.config_file)
        self.transmission_thread = Transmission(self.mi, self.transmission__doocs_ch)
        self.transmission_thread.start()

        self.energy_axis_thread = EnergyAxisWatcher(self.mi, self.ph_energy_ch)
        if self.energy_axis_thread.is_online():
            self.energy_axis_thread.start()

    def get_transmission(self):
        if self.ui.sb_transmission_override.isChecked():
            self.ui.sb_transmission.setEnabled(True)
            value = self.ui.sb_transmission.value()
            if value == 0:
                value = 0.0000001
        else:
            value = self.transmission_thread.transmission
            self.ui.sb_transmission.setValue(value)
            self.ui.sb_transmission.setEnabled(False)
        self.transmission_value = value


    def cross_calibrate(self):
        """
        Cross calibrate with spectrum with pulse energy

        :param spectrum: array
        :param transmission: transmission coefficient 0 - 1
        :param pulse_energy: in [uJ]
        :return: calibration coefficient
        """

        if len(self.ave_spectrum) < 3:
            return
        if self.bunch_num_ctrl.get_value() <= 0:
            self.error_box("No Beam")
            return

        if self.ui.combo_hirex.currentText() in ["DUMMY", "DUMMYSASE"]:
            self.pulse_energy = DummyXGM('','').get_value()
        else:
            self.pulse_energy = self.mi.get_value(self.slow_xgm_signal)

        self.get_transmission()
        self.calib_energy_coef = self.spectrometer.cross_calibrate(self.ave_spectrum, self.x_axis_disp, self.transmission_value, self.pulse_energy)

    def run_settings_window(self):
        if self.settings is None:
            self.settings = HirexSettings(parent=self)
        self.settings.show()

    def run_calculator_window(self):
        if self.calculator_window is None:
            self.calculator_window = UICalculator(parent=self)
        self.calculator_window.show()

    def fit_guass(self):
        if len(self.ave_spectrum) == 0:
            self.error_box("Press Start first")
            return
        try:
            mu = self.spectrometer.fit_guass(self.ave_spectrum) + self.px_first
        except:
            self.error_box("Fitting: Optimal parameters not found")
        self.ui.sb_px1.setValue(mu)

        print("A, mu, sigma = ", self.spectrometer.gauss_coeff_fit)

    def show_fit(self):
        if self.spectrometer.gauss_coeff_fit is None:
            self.error_box("Estimate Px1 first")
            return

        def gauss(x, *p):
            A, mu, sigma = p
            return A*np.exp(-(x-mu)**2/(2.*sigma**2))

        if self.ui.chb_show_fit.isChecked():
            self.plot1.addItem(self.fit_func)
            # self.plot1.setLabel('left', "A", units='au')
            # self.plot1.setLabel('bottom', "", units='px')
            self.x_axis = np.arange(len(self.spectrum_event))
            gauss_coeff_fit = self.spectrometer.gauss_coeff_fit + np.array([0, self.px_first, 0])
            gauss_fit = gauss(self.x_axis, *gauss_coeff_fit)
            self.fit_func.setData(self.x_axis[self.px_first: self.px_last], gauss_fit[self.px_first: self.px_last])
            #self.plot1.enableAutoScale()
            self.plot1.enableAutoRange()
        else:
            self.calibrate_axis()
            self.plot1.removeItem(self.fit_func)
            self.plot1.legend.removeItem(self.fit_func.name())
            # if self.ui.chb_uj_ev.isChecked():
                # self.plot1.setLabel('left', "Spec. density", units='uJ/eV')
            # else:
                # self.plot1.setLabel('left', "A", units='au')
            self.plot1.setLabel('bottom', "", units='eV')
            #self.plot1.enableAutoScale()
            self.plot1.enableAutoRange()

        self.set_labels()

    def set_labels(self):
        if self.ui.chb_show_fit.isChecked():
            self.plot1.setLabel('bottom', "", units='px')
        else:
            self.plot1.setLabel('bottom', "", units='eV')
        if self.ui.chb_uj_ev.isChecked():
                self.plot1.setLabel('left', "Spec. density", units='uJ/eV')
        else:
            self.plot1.setLabel('left', "A", units='au')

    def calibrate_axis(self):
        ev_px = self.ui.sb_ev_px.value()
        E0 = self.ui.sb_E0.value()
        px1 = self.ui.sb_px1.value()
        try:
            self.x_axis = self.spectrometer.calibrate_axis(ev_px, E0, px1)
        except:
            self.error_box("WRONG channel or Device is not available")
            self.x_axis = np.arange(self.hrx_n_px)
        self.reset_waterfall()

    def is_back_taker_alive(self):
        """
        Method to check if the ResponseMatrixCalculator thread is alive.
        it is needed to change name and color of the pushBatton pb_calc_RM.
        When RMs caclulation is finished. If the thread is dead QTimer self.rm_calc is stopped
        :return:
        """
        if not self.back_taker.is_alive():
            self.ui.pb_background.setStyleSheet("color: rgb(0, 0, 0);")
            self.ui.pb_background.setText("Take Background")
            self.background = self.back_taker.background
            self.back_taker_status.stop()
            if self.actual_n_bunchs != self.bunch_num_ctrl.get_value():
                self.bunch_num_ctrl.set_value(self.actual_n_bunchs)

    def take_background(self):
        current_source = self.ui.combo_hirex.currentText()
        if self.ui.pb_background.text() == "Taking ...              ":
            self.ui.pb_background.setStyleSheet("color: rgb(85, 255, 255);")
            self.ui.pb_background.setText("Take Background")
            if self.back_taker.is_alive():
                self.back_taker.stop()
                if self.actual_n_bunchs != self.bunch_num_ctrl.get_value():
                    self.bunch_num_ctrl.set_value(self.actual_n_bunchs)
        else:
            if self.ui.pb_start.text() == "Start":
                self.error_box("Start HIREX first")
                return
            self.actual_n_bunchs = self.bunch_num_ctrl.get_value()
            if self.actual_n_bunchs != 0:
                try:
                    self.bunch_num_ctrl.set_value(0)
                except:
                    self.error_box("No permission. Set to Zero bunches before taking background")
                    return

            self.back_taker = Background(mi=self.mi, device=self.spectrometer, dev_name=current_source)
            self.back_taker.devmode = self.dev_mode


            time.sleep(0.5)
            self.back_taker.nshots = self.sb_nbunch_back
            # self.back_taker.doocs_channel = str(self.ui.le_a.text())
            if not self.back_taker.is_alive():
                self.back_taker.start()
                self.back_taker_status.start()
            self.ui.pb_background.setText("Taking ...              ")
            self.ui.pb_background.setStyleSheet("color: rgb(85, 255, 127);")

    def calc_spec(self):
        self.spectrum_event = self.spectrometer.get_value().astype("float64")
        self.spectrum_event_disp = self.spectrum_event[self.px_first:self.px_last]
        self.x_axis_disp = self.x_axis[self.px_first: self.px_last]
        self.background_disp = self.background[self.px_first:self.px_last]
        if self.ui.chb_a.isChecked():
            if len(self.background_disp) != len(self.spectrum_event_disp):
                self.background_disp = np.zeros_like(self.spectrum_event_disp)
                self.error_box("Take Background")
                self.ui.chb_a.setChecked(False)
            else:
                self.spectrum_event_disp -= self.background_disp

        # send maximum to doocs
        #self.mi.set_value("XFEL_SIM.UTIL/BIG_BROTHER/SASE1_2.3A/Z_POS", np.max(spectrum[350:450]))
        #self.mi.set_value("XFEL_SIM.UTIL/BIG_BROTHER/MAIN/Z_POS", np.sum(spectrum))

        self.spectrum_list.insert(0, self.spectrum_event_disp)
        # self.ave_spectrum = np.mean(self.spectrum_list, axis=0)
        n_av = int(self.ui.sb_av_nbunch.value())
        if len(self.spectrum_list) > n_av:
            self.spectrum_list = self.spectrum_list[:n_av]
        self.ave_spectrum = np.mean(self.spectrum_list, axis=0)

        if not old_scipy:
            filtr_av_spectrum = ndimage.gaussian_filter(self.ave_spectrum, sigma=self.ui.sb_gauss_filter.value())
            peaks, _ = find_peaks(filtr_av_spectrum,  distance=self.ui.sb_mkn_dist_peaks.value(),
                               height=np.max(filtr_av_spectrum)*self.ui.sb_low_thresh.value()/100.,
                              #prominence=0.5
                              )
            self.peak_ev_list = self.x_axis_disp[peaks]
        else:
            self.peak_ev_list = [self.x_axis_disp[np.argmax(self.ave_spectrum)]]
        #print(self.peak_ev_list)

        # single_integr = np.trapz(spectrum, self.x_axis_disp)/self.get_transmission() * self.calib_energy_coef
        self.ave_integ = np.trapz(self.ave_spectrum, self.x_axis_disp) / self.transmission_value * self.calib_energy_coef
        if self.doocs_permit and self.ui.cb_doocs_send_data.isChecked():
            self.mi.set_value(self.dynprop_integ, self.ave_integ)
            self.mi.set_value(self.dynprop_max, np.max(self.ave_spectrum[self.max_spec_min_inx: self.max_spec_max_inx]))
        #self.mi.set_value("XFEL_SIM.UTIL/BIG_BROTHER/MAIN/Z_POS", np.max(self.ave_spectrum[570:580]))
        self.peak_ev = self.x_axis_disp[np.argmax(self.ave_spectrum)]

        self.data_2d = np.roll(self.data_2d, 1, axis=1)

        self.data_2d[:, 0] = self.spectrum_event_disp# single_sig_wo_noise

        try:
            p1interp, p2interp = fwhm3(np.array(self.ave_spectrum))
            fwhm_px = p2interp - p1interp
            peak_px = (p2interp + p1interp)/2
        except:
            fwhm_px = 0
            peak_px = 0
        px_ev = (self.x_axis_disp[1] - self.x_axis_disp[0])
        #print('peak_px',peak_px)
        try:
            peak_px_int = int(np.floor(peak_px))
            self.peak_ev = self.x_axis_disp[peak_px_int] + px_ev * (peak_px - np.floor(peak_px))
        except:
            self.peak_ev = np.nan
        self.fwhm_ev = fwhm_px * px_ev

    def plot_spec(self):
        if self.energy_axis_thread.trigger:
            self.calibrate_axis()
            self.reset_spectrum()
            return
        self.pulse_energy = self.xgm.get_value()
        
        #print('self.spectrum_event_disp.shape',self.spectrum_event_disp.shape)
        #print('self.x_axis_disp.shape',self.x_axis_disp.shape)
        if len(self.spectrum_event_disp) != len(self.x_axis_disp) or len(self.ave_spectrum) != len(self.x_axis_disp):
            return
        
            # if tab is not active plotting paused
        if self.ui.scan_tab.currentIndex() == 0:
            if self.ui.chb_uj_ev.isChecked():
                transm = self.transmission_value
                self.single.setData(x=self.x_axis_disp, y=self.spectrum_event_disp * self.calib_energy_coef / transm)
                self.average.setData(x=self.x_axis_disp, y=self.ave_spectrum * self.calib_energy_coef / transm)
            else:
                self.single.setData(x=self.x_axis_disp, y=self.spectrum_event_disp)
                self.average.setData(x=self.x_axis_disp, y=self.ave_spectrum)
            n_ppoints = len(self.x_axis_disp)
            if len(self.background_disp) != n_ppoints:
                self.background_disp = np.zeros(n_ppoints)
            self.back_plot.setData(self.x_axis_disp, self.background_disp)
            self.img.setImage(self.data_2d) #SS: do not cut, limits window
        #except:
            #print("could not plot spectra, hirex.py -> plot_spec")
            #pass
            # self.average.setData(x=self.x_axis, y=filtr_av_spectrum)
        if self.counter_spect % 10 == 1:
            if np.abs(self.ave_integ-self.pulse_energy)/self.pulse_energy > 0.2:
                integral_text_color='red'
            elif np.abs(self.ave_integ-self.pulse_energy)/self.pulse_energy > 0.05:
                integral_text_color='orange'
            else:
                integral_text_color='green'
            
            self.label2.setText(
            # "<span style='font-size: 16pt', style='color: green'>XGM: %0.2f &mu;J <span style='color: red'>SPEC.INTEGRAL: %0.2f &mu;J   <span style='color: green'> @ %0.1f eV</span>"%(
            "<span style='font-size: 15pt', style='color: blue'>XGM: %0.2f &mu;J   <span style='color: %s'>SPEC.INTEGRAL: %0.2f &mu;J</span>"%(
            self.pulse_energy, integral_text_color, self.ave_integ))
            # try:
            # print('self.x_axis_disp = {}'.format(self.x_axis_disp))

            self.ui.label_sigma.setText(str(np.round(self.fwhm_ev, 3)))
            self.ui.label_peak_ev.setText(str(np.round(self.peak_ev, 3)))
            self.ui.rel_width.setText(str(np.round((self.fwhm_ev/self.peak_ev)*1e2, 3)))
            # print('self.counter_spect {}'.format(self.counter_spect))
            # print('peak_ave {}'.format(np.nanmax(self.ave_spectrum)))
            # print('peak at {} with fwhm of {}'.format(peak_ev, fwhm_ev))
        self.counter_spect += 1

    def sigma_gauss_fit(self, y):
        x = np.arange(len(y))

        def gauss(x, *p):
            A, mu, sigma = p
            return A * np.exp(-(x - mu) ** 2 / (2. * sigma ** 2))

        # (A, mu, sigma)
        p0 = [np.max(y), np.argmax(y), 30]
        try:
            gauss_coeff_fit, var_matrix = curve_fit(gauss, x, y, p0=p0)
        except:
            return 0
        sigma = gauss_coeff_fit[2]
        return sigma



    def show_hide_background(self):
        if self.ui.pb_hide_show_backplot.text() == "Hide Background":
            self.plot1.removeItem(self.back_plot)
            self.plot1.legend.removeItem(self.back_plot.name())
            #self.ui.pb_hide_show_backplot.setStyleSheet("color: rgb(85, 255, 255);")
            self.ui.pb_hide_show_backplot.setText("Show Background")
        else:
            self.ui.pb_hide_show_backplot.setText("Hide Background")
            #self.ui.pb_hide_show_backplot.setStyleSheet("color: rgb(255, 0, 0);")
            self.plot1.addItem(self.back_plot)

    def show_hide_average(self):
        if self.ui.pb_hide_average.text() == "Hide Average":
            self.plot1.removeItem(self.average)
            self.plot1.legend.removeItem(self.average.name())
            self.ui.pb_hide_average.setText("Show Average")
        else:
            self.ui.pb_hide_average.setText("Hide Average")
            self.plot1.addItem(self.average)


    def reset_waterfall(self):

        px_first = self.ui.sb_px_first.value()
        px_last = self.ui.sb_px_last.value()
        num_px = self.spectrometer.num_px - px_first + px_last
        if num_px < 10:
            self.ui.sb_px_first.setValue(self.px_first)
            self.px_last = self.px_last if self.px_last is not None else 0
            self.ui.sb_px_last.setValue(self.px_last)
        else:
            self.px_first = px_first
            self.px_last = px_last
        self.data_2d = np.zeros((self.spectrometer.num_px - self.px_first + self.px_last, int(self.sb_2d_hist_size)))
        self.spectrum_list = []
        self.ave_spectrum = []
        self.px_last = self.px_last if self.px_last != 0 else None


        #self.data_2d = np.zeros((self.spectrometer.num_px, self.sb_2d_hist_size))
        self.x_axis_disp = self.x_axis[self.px_first: self.px_last]
        scale_coef_xaxis = (self.x_axis_disp[-1] - self.x_axis_disp[0]) / len(self.x_axis_disp)
        translate_coef_xaxis = self.x_axis_disp[0] / scale_coef_xaxis
        self.add_image_item()

        self.img.scale(scale_coef_xaxis, 1)
        self.img.translate(translate_coef_xaxis, 0)

    def reset_spectrum(self):
        self.counter_spect = 0
        #self.data_2d = np.zeros((self.spectrometer.num_px, self.sb_2d_hist_size))
        self.spectrum_list = []
        self.ave_spectrum = []

    def start_stop_live_spectrum(self):
        if self.ui.pb_start.text() == "Stop":
            self.timer_live.stop()
            self.timer_plot.stop()
            self.ui.pb_start.setStyleSheet("color: rgb(255, 0, 0); font-size: 18pt")
            self.ui.pb_start.setText("Start")
        else:
            if self.bunch_num_ctrl.get_value() <= 0:
                self.error_box("No Beam. It can cause some problems")
                #return
            if not self.spectrometer.is_online():
                self.error_box("Spectrometer is not ONLINE")
                return
            self.calibrate_axis()
            self.reset_spectrum()
            self.timer_live.start(100)
            self.timer_plot.start(200)
            self.ui.pb_start.setText("Stop")

            #self.ui.pb_start.setStyleSheet("color: rgb(85, 255, 127); font-size: 18pt")
            self.ui.pb_start.setStyleSheet("color: rgb(63, 191, 95); font-size: 18pt")

            #px1 = int(self.ui.sb_px1.value())
            self.reset_waterfall()


    def update_text(self, text=None):
        x_left = self.plot1.viewRange()[0][0]
        x_right = self.plot1.viewRange()[0][1]
        y_down = self.plot1.viewRange()[1][0]
        y_up = self.plot1.viewRange()[1][1]
        x = x_left + (x_right - x_left)*0.7
        y = y_down + (y_up - y_down)*0.9

        #print(self.plot1.viewRange())
        self.textItem.setText(text)
        self.textItem.setPos(x, y)

    def closeEvent(self, event):
        if self.logger_window is not None:
            self.logger_window.close()

        if self.transmission_thread.is_alive():
            self.transmission_thread.kill = True
            self.transmission_thread.stop()

        if self.energy_axis_thread.is_alive():
            self.energy_axis_thread.kill = True
            self.energy_axis_thread.stop()

        if self.timer_live.isActive():
            print("stop live spectrum")
            self.timer_live.stop()

        if self.scantool.scanning is not None:
            self.scantool.scanning.kill = True
            self.scantool.scanning.crystal.stop()

        if self.corre2dtool is not None:
            self.corre2dtool.stop_timers()

        if self.analysistool is not None:
            self.analysistool.stop_timers()

        if 1:
            self.ui.save_state(self.config_file)
        logger.info("close")
        event.accept()  # let the window close

    def parse_arguments(self):
        parser = argparse.ArgumentParser(description="PySpectrometer",
                                         add_help=False)
        parser.set_defaults(mi='XFELMachineInterface')
        parser.add_argument('--devmode', action='store_true',
                            help='Enable development mode.', default=False)

        parser_mi = argparse.ArgumentParser()

        subparser = parser_mi.add_subparsers(title='Machine Interface Options', dest="mi")
        for mi in AVAILABLE_MACHINE_INTERFACES:
            mi_parser = subparser.add_parser(mi.__name__, help='{} arguments'.format(mi.__name__))
            mi.add_args(mi_parser)

        for hirex in AVAILABLE_SPECTROMETERS:
            hirex_agr = "--" + hirex
            parser.add_argument(hirex_agr, hirex_agr.upper(), help=hirex.upper() + " HIREX", action="store_true")

        self.tool_args, others = parser.parse_known_args()
        if len(others) != 0:
            self.tool_args = parser_mi.parse_args(others, namespace=self.tool_args)

    def add_plot(self):
        gui_index = self.ui.get_style_name_index()
        if "standard" in self.gui_styles[gui_index]:
            pg.setConfigOption('background', 'w')
            pg.setConfigOption('foreground', 'k')
            single_pen = pg.mkPen("k")
        else:
            single_pen = pg.mkPen("w")

        win = pg.GraphicsLayoutWidget()
        #justify='right',,
        self.label = pg.LabelItem(justify='left', row=0, col=0)
        win.addItem(self.label)


        #self.plot1 = win.addPlot(row=0, col=0)
        self.plot1 = win.addPlot(row=1, col=0)

        self.label2 = pg.LabelItem( justify='right')
        win.addItem(self.label2, row=0, col=0)

        self.plot1.setLabel('left', "A", units='au')
        self.plot1.setLabel('bottom', "", units='eV')

        self.plot1.showGrid(1, 1, 1)

        self.plot1.getAxis('left').enableAutoSIPrefix(enable=False)  # stop the auto unit scaling on y axes
        layout = QtGui.QGridLayout()
        self.ui.widget.setLayout(layout)
        layout.addWidget(win, 0, 0)

        self.plot1.setAutoVisible(y=True)

        self.plot1.addLegend()

        self.single = pg.PlotCurveItem(pen=single_pen, name='single')

        self.plot1.addItem(self.single)

        pen = pg.mkPen((51, 255, 51), width=2)
        pen = pg.mkPen((255, 0, 0), width=3)
        #self.average = pg.PlotCurveItem(x=[], y=[], pen=pen, name='average')
        self.average = pg.PlotCurveItem( pen=pen, name='average')

        self.plot1.addItem(self.average)

        pen = pg.mkPen((0, 255, 255), width=2)

        self.fit_func = pg.PlotCurveItem(pen=pen, name='Gauss Fit')

        #self.plot1.addItem(self.fit_func)
        #self.plot1.enableAutoRange(False)
        #self.textItem = pg.TextItem(text="", border='w', fill=(0, 0, 0))
        # self.textItem.setPos(10, 10)

        pen = pg.mkPen((0, 100, 0), width=1)
        #self.average = pg.PlotCurveItem(x=[], y=[], pen=pen, name='average')
        self.back_plot = pg.PlotCurveItem( pen=pen, name='background')

        #self.plot1.addItem(self.back_plot) ##################################### SS removed, as typically we don;t need it once start pySpectrometer

        # cross hair
        self.vLine = pg.InfiniteLine(angle=90, movable=False)
        self.hLine = pg.InfiniteLine(angle=0, movable=False)
        self.plot1.addItem(self.vLine, ignoreBounds=True)
        self.plot1.addItem(self.hLine, ignoreBounds=True)

        #self.plot1.sigRangeChanged.connect(self.zoom_signal)

    def mouseMoved(self, evt):
        #print("here", evt)
        #pos = evt.x(), evt.y() #evt[0]  ## using signal proxy turns original arguments into a tuple
        #print(evt.x())
        if self.plot1.sceneBoundingRect().contains(evt.x(), evt.y()):
            mousePoint = self.plot1.vb.mapSceneToView(evt)
            # index = int(mousePoint.x())
            array = np.asarray(self.x_axis)
            index = (np.abs(array - mousePoint.x())).argmin()
            if index > 0 and index < len(self.x_axis):
                self.label.setText(
                    "<span style='font-size: 10pt', style='color: green'>x=%0.1f,   <span style='color: red'>y=%0.1f</span>" % (
                    mousePoint.x(), mousePoint.y()))
            self.vLine.setPos(mousePoint.x())
            self.hLine.setPos(mousePoint.y())

    def add_image_widget(self):
        win = pg.GraphicsLayoutWidget()
        layout = QtGui.QGridLayout()
        self.ui.widget_2.setLayout(layout)
        layout.addWidget(win)

        self.img_plot = win.addPlot()
        self.add_image_item()

    def add_image_item(self):
        self.img_plot.clear()

        self.img_plot.setLabel('left', "N bunch", units='')
        self.img_plot.setLabel('bottom', "", units='eV')

        self.img = pg.ImageItem()

        self.img_plot.addItem(self.img)

        colormap = cm.get_cmap('viridis') #"nipy_spectral")  # cm.get_cmap("CMRmap")
        colormap._init()
        lut = (colormap._lut * 255).view(np.ndarray)  # Convert matplotlib colormap from 0-1 to 0 -255 for Qt

        # Apply the colormap
        self.img.setLookupTable(lut)


    def error_box(self, message):
        QtGui.QMessageBox.about(self, "Error box", message)

    def question_box(self, message):
        #QtGui.QMessageBox.question(self, "Question box", message)
        reply = QtGui.QMessageBox.question(self, "Recalculate ORM?",
                "Recalculate Orbit Response Matrix?",
                QtGui.QMessageBox.Yes | QtGui.QMessageBox.No)
        if reply == QtGui.QMessageBox.Yes:
            return True

        return False

    def zoom_signal(self):
        #s = self.plot1.viewRange()[0][0]

        s_up = self.plot1.viewRange()[0][0]
        s_down = self.plot1.viewRange()[0][1]
        #print(np.argwhere(self.x_axis > s_up))
        #print(np.argwhere(self.x_axis < s_down))
        #print(f"s_down = {s_down}, s_up = {s_up}, min(axis) = {np.min(self.x_axis)}, max(axis) = {np.max(self.x_axis)}")
        #indx1 = np.argwhere(self.x_axis > s_up)[0]
        #indx2 = np.argwhere(self.x_axis < s_down)[-1]

        #print(s_up, s_down, indx1, indx2)
        #s_up = s_up if s_up <= s_pos[-1] else s_pos[-1]
        #s_down = s_down if s_down >= s_pos[0] else s_pos[0]

    def load_settings(self):
        logger.debug("load settings ... ")
        with open(self.settings_file, 'r') as f:
            table = json.load(f)
        current_source = self.ui.combo_hirex.currentText()
        self.hrx_n_px = 1000
        self.ph_energy_ch = None
        self.sb_nbunch_back = table["sb_nbunch_back"]
        if current_source == "SASE2":

            self.hirex_doocs_ch = table["le_hirex_ch_sa2"]
            self.transmission__doocs_ch = table["le_trans_ch_sa2"]
            self.hrx_n_px = table["sb_hrx_npx_sa2"]

            self.doocs_ctrl_num_bunch = table["le_ctrl_num_bunch_sa2"]
            self.fast_xgm_signal = table["le_fast_xgm_sa2"]
            self.slow_xgm_signal = table["le_slow_xgm_sa2"]

        elif current_source == "SASE1":
            self.hirex_doocs_ch = table["le_hirex_ch_sa1"]
            self.transmission__doocs_ch = table["le_trans_ch_sa1"]
            self.hrx_n_px = table["sb_hrx_npx_sa1"]

            self.doocs_ctrl_num_bunch = table["le_ctrl_num_bunch_sa1"]
            self.fast_xgm_signal = table["le_fast_xgm_sa1"]
            self.slow_xgm_signal = table["le_slow_xgm_sa1"]

        elif current_source in ["SASE3", "SASE3_SQS"]:
            self.hirex_doocs_ch = table["le_hirex_ch_sa3"]
            self.ph_energy_ch = table["le_ph_energy_sa3"]
            self.transmission__doocs_ch = table["le_trans_ch_sa3"]
            print("self.transmission__doocs_ch", self.transmission__doocs_ch)
            self.hrx_n_px = table["sb_hrx_npx_sa3"]

            self.doocs_ctrl_num_bunch = table["le_ctrl_num_bunch_sa3"]
            self.fast_xgm_signal = table["le_fast_xgm_sa3"]
            self.slow_xgm_signal = table["le_slow_xgm_sa3"]

        elif current_source == "VIKING":
            self.hirex_doocs_ch = 'XFEL.EXP/MDL.EXP_SPECTROMETER/SCS_EXP_VIKING/INTENSITYDISTRIBUTION'
            self.ph_energy_ch = 'XFEL.EXP/MDL.EXP_SPECTROMETER/SCS_EXP_VIKING/PHOTONENERGY'
            self.transmission__doocs_ch = table["le_trans_ch_sa3"]
            print("self.transmission__doocs_ch", self.transmission__doocs_ch)
            self.hrx_n_px = table["sb_hrx_npx_sa3"]

            self.doocs_ctrl_num_bunch = table["le_ctrl_num_bunch_sa3"]
            self.fast_xgm_signal = table["le_fast_xgm_sa3"]
            self.slow_xgm_signal = table["le_slow_xgm_sa3"]

        elif current_source == "SASE3_SCS":
            self.hirex_doocs_ch = table["le_hirex_ch_sa3_scs"]
            self.ph_energy_ch = table["le_ph_energy_sa3_scs"]
            self.transmission__doocs_ch = table["le_trans_ch_sa3_scs"]
            print("self.transmission__doocs_ch", self.transmission__doocs_ch)
            self.hrx_n_px = table["sb_hrx_npx_sa3_scs"]

            self.doocs_ctrl_num_bunch = table["le_ctrl_num_bunch_sa3_scs"]
            self.fast_xgm_signal = table["le_fast_xgm_sa3_scs"]
            self.slow_xgm_signal = table["le_slow_xgm_sa3_scs"]


        elif current_source in ["DUMMY", "DUMMYSASE"]:
            self.hirex_doocs_ch = table["le_hirex_ch_sa1"]
            self.transmission__doocs_ch = None
            self.hrx_n_px = 3000

            self.doocs_ctrl_num_bunch = None
            self.fast_xgm_signal = None
            self.slow_xgm_signal = None


        elif current_source in ["TEST1"]:
            self.hirex_doocs_ch = table["le_hirex_ch_test1"]
            self.transmission__doocs_ch = None
            self.hrx_n_px = 3000

            self.doocs_ctrl_num_bunch = None
            self.fast_xgm_signal = None
            self.slow_xgm_signal = None
            self.ph_energy_ch = None
            print("TEST DEVICES")
        self.dynprop_max = table["le_dynprop_max"]
        self.dynprop_integ = table["le_dynprop_integ"]
        self.max_spec_min_inx = table["sb_max_spec_min"]
        self.max_spec_max_inx = table["sb_max_spec_max"]

        if self.max_spec_max_inx > self.hrx_n_px:
            self.max_spec_max_inx = self.hrx_n_px

        if self.max_spec_min_inx >= self.max_spec_max_inx:
            self.max_spec_min_inx = self.max_spec_max_inx - 1

        self.logbook = table["logbook"]
        if "server" in table.keys():
            self.server = table["server"]
        else:
            self.server = "XFEL"

        self.sb_2d_hist_size = table['sb_2d_hist_size']

        logger.debug("load settings ... OK")

    def loadStyleSheet(self):
        """ Sets the dark GUI theme from a css file."""
        try:
            self.cssfile = "gui/style.css"
            with open(self.cssfile, "r") as f:
                self.setStyleSheet(f.read())
        except IOError:
            logger.error('No style sheet found!')


    def save_data_as(self, type):
        filename = QtGui.QFileDialog.getSaveFileName(self, 'Save Data',
                                                     self.data_dir, "txt (*.npz)", None,
                                                     QtGui.QFileDialog.DontUseNativeDialog)[0]

        np.savez(filename, e_axis=self.x_axis, average=self.ave_spectrum, map=self.data_2d)



def main():


    #make pyqt threadsafe
    QtCore.QCoreApplication.setAttribute(QtCore.Qt.AA_X11InitThreads)
    #create the application
    app = QApplication(sys.argv)
    path = os.path.join(os.path.dirname(sys.modules[__name__].__file__), 'gui/hirex.png')
    app.setWindowIcon(QtGui.QIcon(path))

    window = SpectrometerWindow()


    #show app
    #window.setWindowIcon(QtGui.QIcon('gui/angry_manul.png'))
    # setting the path variable for icon
    #path = os.path.join(os.path.dirname(sys.modules[__name__].__file__), 'gui/manul.png')
    #app.setWindowIcon(QtGui.QIcon(path))
    window.show()
    window.raise_()
    #Build documentaiton if source files have changed
    # TODO: make more universal
    #os.system("cd ./docs && xterm -T 'Ocelot Doc Builder' -e 'bash checkDocBuild.sh' &")
    #exit script
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
