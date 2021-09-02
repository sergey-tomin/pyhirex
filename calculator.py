"""
Christian Grech, DESY, 2021
based on logger.py by Sergey Tomin
"""
import sys
import numpy as np
import json
import pathlib
from PyQt5 import QtGui, QtCore
from PyQt5.QtWidgets import QWidget, QMessageBox, QApplication
import pyqtgraph as pg
from gui.UICalculator import Ui_Form
from mint.opt_objects import Device
import time
import os
import glob
from threading import Thread, Event
import logging
from matplotlib import cm
import pandas as pd
from scipy import ndimage
from scipy.spatial import distance
from scipy.optimize import fsolve
from skimage.filters import threshold_yen
from scipy import interpolate
import re
from skimage.transform import hough_line, hough_line_peaks
from model_functions.HXRSS_Bragg_max_generator import HXRSS_Bragg_max_generator
from model_functions.HXRSS_Bragg_generator import HXRSS_Bragg_generator
from itertools import cycle
path = os.path.realpath(__file__)
indx = path.find("hirex.py")
print("PATH to main file: " + os.path.realpath(__file__)
      + " path to folder: " + path[:indx])
sys.path.insert(0, path[:indx])
# filename="logs/afb.log",
#logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

PY_SPECTROMETER_DIR = "pySpectrometer"
DIR_NAME = "hirex"


def find_nearest_idx(array, value):
    idx = np.abs(array - value).argmin()
    return idx


class UICalculator(QWidget):
    def __init__(self, parent=None):
        #QWidget.__init__(self, parent)
        super().__init__()
        self.parent = parent
        self.ui = Ui_Form()
        self.ui.setupUi(self)
        gui_index = self.parent.ui.get_style_name_index()
        style_name = self.parent.gui_styles[gui_index]
        self.loadStyleSheet(filename=self.parent.gui_dir + style_name)
        self.mi = self.parent.mi
        self.decimals_rounding = 4
        self.colors = ['r', 'b', 'g', 'c', 'y', 'k']
        self.colors2 = ['b', 'g', 'c', 'y', 'k']
        self.linecolors = cycle(self.colors)
        self.linecolors1 = cycle(self.colors2)
        self.n = 0
        self.d_kernel = 0
        self.e_kernel = 0
        self.mode = 0
        self.mono_no = None
        self.max_E = 800
        self.max_P = 2
        self.slope_allowance = 3
        self.intercept_allowance = 100
        self.max_distance = 1400
        self.hmax, self.kmax, self.lmax = 5, 5, 5
        self.img_corr2d = None
        self.min_phen = 0
        self.max_phen = 0
        self.min_pangle = 0
        self.max_pangle = 0
        self.dE_mean = 0
        self.nomatch = 0
        self.yvalue = []
        self.spec_hist = []
        self.doocs_vals_hist = []
        self.doocs_address_label = ''
        self.spec_binned = []
        self.doocs_bins = []
        self.doocs_event_counts = []
        self.doocs_vals_hist_lagged = []
        self.pitch_angle_range, self.min_angle_list, self.spec_data_list, self.slope_list, self.y_intercept_list, self.centroid_pa_list, self.centroid_phen_list, self.max_angle_list = [], [], [], [], [], [], [], []
        self.tngnt_slope_list, self.tngnt_intercept_list, self.tngnt_gid_list, self.tngnt_centroid_list, self.tngnt_centroid_y_list, self.tngnt_roll_angle_list, self.interp_Bragg_list = [], [], [], [], [], [], []
        self.detected_slope_list, self.detected_intercept_list, self.detected_id_list, self.detected_line_min_angle_list, self.detected_line_max_angle_list,  self.detected_line_roll_angle_list, self.dE_list, self.ans_list, self.detected_centroid_x_list, self.detected_centroid_y_list = [], [], [], [], [], [], [], [], [], []
        self.h_list, self.k_list, self.l_list, self.roll_list = [], [], [], []
        self.event_counter = 0

        DIR_NAME = os.path.basename(pathlib.Path(__file__).parent.absolute())
        self.path = path[:path.find(DIR_NAME)]
        self.data_dir = path[:path.find(
            "user")] + "user" + os.sep + PY_SPECTROMETER_DIR + os.sep + "SASE2" + os.sep
        print(self.data_dir)

        self.ui.pb_start_calc.clicked.connect(self.start_stop_calc_from_npz)
        self.ui.pb_scan.clicked.connect(self.start_stop_scan)
        self.ui.pb_calculate.clicked.connect(self.start_stop_calc)

        self.ui.browse_button.clicked.connect(self.open_file)
        self.ui.file_name.setText('')
        self.ui.roll_angle.setDecimals(4)
        self.ui.roll_angle.setSuffix(" °")
        self.ui.roll_angle.setRange(0, 2)
        self.ui.roll_angle.setValue(1.5013)
        self.ui.roll_angle.setSingleStep(0.001)
        self.ui.combo_mono.addItem("Monochromator 1")
        self.ui.combo_mono.addItem("Monochromator 2")
        # Set up and show the two graph axes
        self.add_image_widget()
        self.add_plot_widget()
        self.get_latest_npz()

        #self.ui = self.parent.ui

    def reset(self):

        #self.text.setText('')
        self.dE_mean = 0
        self.min_phen = 0
        self.max_phen = 0
        self.min_pangle = 0
        self.max_pangle = 0
        self.dE_mean = 0
        self.pitch_angle_range, self.min_angle_list, self.spec_data_list, self.slope_list, self.y_intercept_list, self.centroid_pa_list, self.centroid_phen_list, self.max_angle_list = [], [], [], [], [], [], [], []
        self.tngnt_slope_list, self.tngnt_intercept_list, self.tngnt_gid_list, self.tngnt_centroid_list, self.tngnt_centroid_y_list, self.tngnt_roll_angle_list, self.interp_Bragg_list = [], [], [], [], [], [], []
        self.detected_slope_list, self.detected_intercept_list, self.detected_id_list, self.detected_line_min_angle_list, self.detected_line_max_angle_list,  self.detected_line_roll_angle_list, self.dE_list, self.ans_list, self.detected_centroid_x_list, self.detected_centroid_y_list = [], [], [], [], [], [], [], [], [], []
        self.h_list, self.k_list, self.l_list, self.roll_list, self.pa, self.phen = [
            ], [], [], [], [], []
        #if self.counter > 0:
        if self.mode == 1:
            self.img_corr2d.clear()
            self.plot1.clear()
            if self.nomatch == 0:
                self.legend.scene().removeItem(self.legend)
                self.model.setData(x=[], y=[])
                self.line.setData(x=[], y=[])
                self.line_shifted.setData(x=[], y=[])
            self.ui.output.setText('')
            self.info_mono_no()
            self.ui.pb_start_calc.setStyleSheet(
                "color: rgb(85, 255, 127); font-size: 14pt")
            self.ui.pb_start_calc.setText("Calculate fom npz file")
            self.mode = 0

        elif self.mode == 2:
            self.spec_hist = []
            self.doocs_vals_hist = []
            self.spec_binned = []
            self.doocs_bins = []
            self.doocs_event_counts = []
            self.doocs_vals_hist_lagged = []
            self.img_corr2d.clear()
            self.plot1.clear()
            self.ui.pb_scan.setStyleSheet(
                "color: rgb(85, 255, 127); font-size: 14pt")
            self.ui.pb_scan.setText("Scan")
            self.ui.pb_scan.setStyleSheet(
                "color: rgb(85, 255, 127); font-size: 14pt")
            self.ui.pb_calculate.setText("Calculate")
            self.mode = 0
        #self.counter = self.counter + 1

        #self.ui.roll_angle.clear()

    def closeEvent(self, QCloseEvent):
    	self.mode = 0
    	self.reset()

    def start_stop_calc_from_npz(self):
        self.mode = 1
        if self.ui.pb_start_calc.text() == "Reset":
            self.reset()
        else:
            if self.ui.mono_no.text() == "Invalid input":
                self.error_box("Select a valid npz file first")
                return
            if self.ui.mono_no.text() == "":
                self.error_box("Select a valid npz file first")
                return
            self.load_corr2d()
            self.corr2d = self.tt['corr2d']
            self.binarization()
            self.ui.mono_no.setText('Image binarization complete')
            self.get_binarized_line()
            self.img_processing()
            self.add_corr2d_image_item()
            self.hough_line_transform()
            self.generate_Bragg_curves()
            self.tangent_generator()
            self.line_comparator()
            if len(self.df_detected.index) != 0:
                self.nomatch = 0
                self.hkl_roll_separator()
            # Get Bragg curves
                self.offset_calc_and_plot()
            # If no lines are detected
            else:
                logger.info('No lines can be matched')
                self.ui.mono_no.setText('No lines can be matched')
                self.nomatch = 1
            self.ui.pb_start_calc.setText("Reset")
            self.ui.pb_start_calc.setStyleSheet(
                "color: rgb(255, 0, 0); font-size: 14pt")

    def start_stop_scan(self):
        self.mode = 2
        if self.ui.pb_scan.text() == "Reset":
            self.reset()
        else:
            if self.parent.ui.pb_start.text() == "Start":
                self.error_box("Start spectrometer first")
                return
            if not self.parent.spectrometer.is_online():
                self.error_box("Spectrometer is not ONLINE")
                return
            self.doocs_dev = None
            self.get_device()
            self.plot_correl_scan()
            self.ui.pb_scan.setText("Reset")

            self.ui.pb_scan.setStyleSheet(
                "color: rgb(255, 0, 0); font-size: 14pt")

    def start_stop_calc(self):
        self.mode = 2
        if self.ui.pb_calculate.text() == "Reset":
            self.reset()
        else:
            if self.parent.ui.pb_start.text() == "Start":
                self.error_box("Start spectrometer first")
                return
            if self.ui.pb_scan.text() == "Scan":
                self.error_box("Start scan first")
                return
            if not self.parent.spectrometer.is_online():
                self.error_box("Spectrometer is not ONLINE")
                return
            self.corr2d = self.orig_image
            self.binarization()
            self.ui.pb_calculate.setText("Reset")

            self.ui.pb_calculate.setStyleSheet(
                "color: rgb(255, 0, 0); font-size: 14pt")

    def add_image_widget(self):
        win = pg.GraphicsLayoutWidget()
        self.layout = QtGui.QGridLayout()
        self.ui.widget_calc.setLayout(self.layout)
        self.layout.addWidget(win)
        self.img_corr2d = win.addPlot()
        self.img_corr2d.setLabel('left', "E_ph", units='eV')
        self.img_corr2d.setLabel('bottom', "Pitch angle", units='°')

    def add_corr2d_image_item(self):
        self.img_corr2d.clear()

        scale_yaxis = (self.np_phen[-1] - self.np_phen[0]) / len(self.np_phen)
        translate_yaxis = self.np_phen[0] / scale_yaxis

        scale_xaxis = (max(self.np_doocs) - min(self.np_doocs)
                       ) / len(self.np_doocs)
        translate_xaxis = min(self.np_doocs) / scale_xaxis

        self.img = pg.ImageItem()
        self.img_corr2d.addItem(self.img)
        colormap = cm.get_cmap('viridis')
        colormap._init()
        # Convert matplotlib colormap from 0-1 to 0 -255 for Qt
        lut = (colormap._lut * 255).view(np.ndarray)
        # Apply the colormap
        self.img.setLookupTable(lut)
        self.img.setImage(self.orig_image)
        self.img.scale(scale_xaxis, scale_yaxis)
        self.img.translate(translate_xaxis, translate_yaxis)

    def add_plot_widget(self):
        gui_index = self.parent.ui.get_style_name_index()
        if "standard" in self.parent.gui_styles[gui_index]:
            pg.setConfigOption('background', 'w')
            pg.setConfigOption('foreground', 'k')
            model_pen = pg.mkPen("k")
        else:
            model_pen = pg.mkPen("w")

        win = pg.GraphicsLayoutWidget()
        self.label = pg.LabelItem(justify='left', row=0, col=0)
        win.addItem(self.label)

        self.plot1 = win.addPlot(row=1, col=0)
        self.plot1.setLabel('left', "E_ph", units='eV')
        self.plot1.setLabel('bottom', "Pitch angle", units='°')
        self.plot1.showGrid(1, 1, 1)
        self.plot1.getAxis('left').enableAutoSIPrefix(
            enable=False)  # stop the auto unit scaling on y axes
        self.layout_2 = QtGui.QGridLayout()
        self.ui.widget_calc_2.setLayout(self.layout_2)
        self.layout_2.addWidget(win, 0, 0)
        self.plot1.setAutoVisible(y=True)

        # cross hair
        self.vLine = pg.InfiniteLine(angle=90, movable=False)
        self.hLine = pg.InfiniteLine(angle=0, movable=False)
        self.plot1.addItem(self.vLine, ignoreBounds=True)
        self.plot1.addItem(self.hLine, ignoreBounds=True)

    def add_plot(self):
        #self.plot1.clear()
        self.plot1.enableAutoRange()
        pen = pg.mkPen('r', width=3)
        pen_shifted = pg.mkPen('k', width=3, style=QtCore.Qt.DashLine)
        self.legend = self.plot1.addLegend()
        #self.legend_boolean = 1
        for r in range(len(self.pa)):
            self.model = pg.PlotCurveItem(
                x=self.pa[r], y=self.phen[r], pen=pen)
            self.plot1.addItem(self.model)

        for slope, intercept, min_angle, max_angle, gid in zip(self.df_detected['slope'], self.df_detected['intercept'], self.df_detected['min_angle'], self.df_detected['max_angle'], self.df_detected['gid']):
            pitch_angle_range = np.linspace(min_angle, max_angle, 100)
            self.yvalue = (pitch_angle_range*slope)+intercept
            line_pen = pg.mkPen(next(self.linecolors1), width=3,
                                style=QtCore.Qt.DashLine)
            self.line = pg.PlotCurveItem(
                x=pitch_angle_range, y=self.yvalue, pen=line_pen, name=gid)
            self.line_shifted = pg.PlotCurveItem(x=pitch_angle_range, y=self.yvalue+self.dE_mean,
                                                 pen=pen_shifted)
            self.plot1.addItem(self.line)
            self.plot1.addItem(self.line_shifted)
            #self.change_label(gid)

    def binarization(self):
        # all values below 0 threshold are set to 0
        self.phen_res = self.np_phen[2] - self.np_phen[1]
        self.angle_res = self.np_doocs[2] - self.np_doocs[1]
        self.min_pangle = min(self.np_doocs)
        self.max_pangle = max(self.np_doocs)
        self.corr2d[self.corr2d < 0] = 0
        # define parameters for binarization
        #range_scale = np.ptp(self.corr2d)
        #threshold = self.thresh * range_scale
        #max_value = np.amax(self.corr2d)
        #min_value = np.amin(self.corr2d)
        # all values above threshold are set to max_value
        #self.corr2d[self.corr2d > threshold] = max_value
        # all values above threshold are set to min_value
        #self.corr2d[self.corr2d < threshold] = min_value
        #self.processed_image = self.corr2d.T
        self.image = self.corr2d.T
        thresh = threshold_yen(self.image, nbins=256)
        binary = self.image > thresh
        self.processed_image = binary

    def get_binarized_line(self):
        df = pd.DataFrame(data=self.processed_image.T)
        df_scale = pd.DataFrame(data=self.np_doocs)
        df_scale.columns = ['parameter']
        df_phen = pd.DataFrame(data=self.np_phen)
        df_phen.columns = ['value']
        df_phen = df_phen.T
        df = df.append(df_phen)
        df.columns = df.iloc[-1]
        df.drop(df.tail(1).index, inplace=True)
        df = df.join(df_scale, lsuffix='caller', rsuffix='other')
        df.set_index('parameter', inplace=True)
        df1 = df.stack().reset_index()
        #set column names
        df1.columns = ['Parameter', 'Energy', 'Correlation']
        self.df2 = df1[df1['Correlation'] != False]
        self.df2 = self.df2.drop(columns=['Correlation'])
        self.min_phen = min(self.df2['Energy'])
        self.max_phen = max(self.df2['Energy'])

    def hough_line_transform(self):
        # Classic straight-line Hough transform .accessibleDescription Set a precision of 0.5 degree.
        tested_angles = np.linspace(-np.pi/2, np.pi/2, 360, endpoint=False)
        h, theta, d = hough_line(self.processed_image, theta=tested_angles)
        _, pitch_angle_list, rho_list = hough_line_peaks(
            h, theta, d, num_peaks=5, min_distance=30, min_angle=30)
        if len(pitch_angle_list) == 0:
            self.ui.mono_no.setText('No lines detected')
        else:
            self.ui.mono_no.setText('%d line(s) found' % len(pitch_angle_list))
        for pitch_angle, rho in zip(pitch_angle_list, rho_list):

            # Calculate slope and intercept
            y_intercept = min(self.np_phen) + (rho*self.phen_res/np.sin(pitch_angle))+(
                self.min_pangle*self.phen_res*np.cos(pitch_angle)/(self.angle_res*np.sin(pitch_angle)))
            slope = -(self.phen_res*np.cos(pitch_angle)
                      / (self.angle_res*np.sin(pitch_angle)))
            # Inverse calculation of the pitch angle based on the energy range of the spectrometer data
            pa_1 = (self.min_phen-y_intercept)/slope
            pa_2 = (self.max_phen-y_intercept)/slope
            # Assign max or min angle status based on the polarity of the slope
            if pa_1 < pa_2:
                min_line_pangle = pa_1
                max_line_pangle = pa_2
            else:
                min_line_pangle = pa_2
                max_line_pangle = pa_1
            if self.min_pangle > min_line_pangle:
                min_line_pangle = self.min_pangle
            if self.max_pangle < max_line_pangle:
                max_line_pangle = self.max_pangle
            pa_vec = [pa_1, pa_2]
            phen_vec = [self.min_phen, self.max_phen]
            centroid_pa = np.mean(pa_vec)
            centroid_phen = np.mean(phen_vec)

            # ignore lines which are horizontal
            if slope <= 5 and slope >= -5:
                continue
            line_range = np.linspace(min_line_pangle, max_line_pangle, 10)
            pen = pg.mkPen('r', width=4,
                           style=QtCore.Qt.DashLine)
            self.plt = pg.PlotCurveItem(
                line_range, (slope*line_range) + y_intercept, pen=pen)
            self.img_corr2d.addItem(self.plt)
            self.slope_list.append(slope)
            self.y_intercept_list.append(y_intercept)
            self.centroid_pa_list.append(centroid_pa)
            self.centroid_phen_list.append(centroid_phen)
            self.min_angle_list.append(min_line_pangle)
            self.max_angle_list.append(max_line_pangle)

        self.df_spec_lines = pd.DataFrame(dict(slope=self.slope_list, intercept=self.y_intercept_list, min_angle=self.min_angle_list, max_angle=self.max_angle_list,
                                               centroid_pa=self.centroid_pa_list, centroid_phen=self.centroid_phen_list))
        self.df_spec_lines['roll_angle'] = self.ui.roll_angle.value()

    def generate_Bragg_curves(self):
        self.roll = list(self.df_spec_lines['roll_angle'])
        if self.mono_no == 2:
            self.DTHP = -0.38565
            self.dthy = 1.17
            self.DTHR = 0.1675
            self.alpha = 0.00238
        else:
            self.DTHP = -0.38565
            self.dthy = 1.17
            self.DTHR = 0.1675
            self.alpha = 0.00238
        self.pa_range = np.linspace(self.min_pangle-1, self.max_pangle+1, 200)
        # pass pitch and roll errors and create Bragg curves
        self.phen_list, self.p_angle_list, self.gid_list, self.roll_angle_list = HXRSS_Bragg_max_generator(
            self.pa_range, self.hmax, self.kmax, self.lmax, self.DTHP, self.dthy, self.roll, self.DTHR, self.alpha)
        logger.info("Bragg lines generated")

    def tangent_generator(self):
        for r, gid_raw, roll_angle in zip(range(len(self.p_angle_list)), self.gid_list, self.roll_angle_list):
            x = np.asarray(self.p_angle_list[r])
            y = np.asarray(self.phen_list[r])
            # Interpolating range
            x0 = np.linspace(min(self.p_angle_list[r]), max(
                self.p_angle_list[r]), 200, endpoint=False)

            gid = str(gid_raw)
            f = interpolate.UnivariateSpline(
                self.p_angle_list[r], self.phen_list[r])
            for x0_ in x0:
                i0 = np.argmin(np.abs(x-x0_))
                x1 = x[i0:i0+2]
                y1 = y[i0:i0+2]
                dydx, = np.diff(y1)/np.diff(x1)
                if y1[0] < max(self.np_phen)+500 and y1[0] > min(self.np_phen)-500:
                    def tngnt(x): return dydx*x + (y1[0]-dydx*x1[0])
                    tngnt_slope = (tngnt(x[1])-tngnt(x[0]))/(x[1]-x[0])
                    self.tngnt_slope_list.append(tngnt_slope)
                    tngnt_intercept = tngnt(0)
                    self.tngnt_intercept_list.append(tngnt_intercept)
                    self.tngnt_centroid_list.append(x1[0])
                    self.tngnt_centroid_y_list.append(y1[0])
                    self.tngnt_gid_list.append(gid)
                    self.tngnt_roll_angle_list.append(roll_angle)
                    self.interp_Bragg_list.append(f)
        self.df_tangents = pd.DataFrame(dict(slope=self.tngnt_slope_list, intercept=self.tngnt_intercept_list, gid=self.tngnt_gid_list, interp=self.interp_Bragg_list,
                                             centroid_pa=self.tngnt_centroid_list, centroid_phen=self.tngnt_centroid_y_list, roll_angle=self.tngnt_roll_angle_list))
        logger.info("Tangents generated")

    def line_comparator(self):
        for slope, intercept, min_angle, max_angle, centroid_pa, centroid_phen, roll_angle in zip(self.df_spec_lines['slope'], self.df_spec_lines['intercept'], self.df_spec_lines['min_angle'], self.df_spec_lines['max_angle'], self.df_spec_lines['centroid_pa'], self.df_spec_lines['centroid_phen'], self.df_spec_lines['roll_angle']):
            n = 0
            distance_list = []
            for tngnt_slope, tngnt_intercept, curve_id, interp_fn_Bragg, centroid, centroid_y in zip(self.df_tangents['slope'], self.df_tangents['intercept'], self.df_tangents['gid'], self.df_tangents['interp'], self.df_tangents['centroid_pa'], self.df_tangents['centroid_phen']):
                if (tngnt_slope-self.slope_allowance <= slope <= tngnt_slope+self.slope_allowance) and (tngnt_intercept-self.intercept_allowance <= intercept <= tngnt_intercept+self.intercept_allowance):
                    a = (centroid, centroid_y)
                    b = (centroid_pa, centroid_phen)
                    dist = distance.euclidean(a, b)
                    distance_list.append(dist)
                    if n >= 1 and distance_list[n] < distance_list[n-1] and dist < self.max_distance:
                        def func(x): return interp_fn_Bragg(
                                x)-centroid_phen
                        ans, = fsolve(func, centroid_pa)
                        dE = (interp_fn_Bragg(centroid_pa)-centroid_phen)
                        dP = (ans-centroid_pa)
                        if abs(dE) < self.max_E and abs(dP) < self.max_P:
                            self.detected_slope_list.pop()
                            self.detected_intercept_list.pop()
                            self.detected_id_list.pop()
                            self.detected_line_min_angle_list.pop()
                            self.detected_line_max_angle_list.pop()
                            self.detected_line_roll_angle_list.pop()
                            self.detected_centroid_x_list.pop()
                            self.detected_centroid_y_list.pop()
                            self.dE_list.pop()
                            self.detected_slope_list.append(slope)
                            self.detected_intercept_list.append(intercept)
                            self.detected_id_list.append(curve_id)
                            self.detected_line_min_angle_list.append(
                                    min_angle)
                            self.detected_line_max_angle_list.append(
                                    max_angle)
                            self.detected_line_roll_angle_list.append(
                                    roll_angle)
                            self.dE_list.append(dE)
                            self.detected_centroid_x_list.append(centroid_pa)
                            self.detected_centroid_y_list.append(centroid_phen)
                            n = n+1
                            #logger.info('Its a match ', n, ' Curve id:', curve_id, 'Distance', np.round(
                            #dist, 2), np.round(ans, 2), np.round(interp_fn_Bragg(ans), 2))
                    elif n >= 1 and distance_list[n] >= distance_list[n-1]:
                        pass
                    elif dist < self.max_distance:
                        def func(x): return interp_fn_Bragg(
                            x)-centroid_phen
                        ans, = fsolve(func, centroid_pa)
                        dE = (interp_fn_Bragg(centroid_pa)-centroid_phen)
                        dP = (ans-centroid_pa)
                        if abs(dE) < self.max_E and abs(dP) < self.max_P:
                            self.detected_slope_list.append(slope)
                            self.detected_intercept_list.append(intercept)
                            self.detected_id_list.append(curve_id)
                            self.detected_line_min_angle_list.append(
                                    min_angle)
                            self.detected_line_max_angle_list.append(
                                    max_angle)
                            self.detected_line_roll_angle_list.append(
                                    roll_angle)
                            self.dE_list.append(dE)
                            self.detected_centroid_x_list.append(centroid_pa)
                            self.detected_centroid_y_list.append(centroid_phen)
                            n = n+1
                            print('Its a match ', n, ' Curve id:', curve_id, 'Distance', np.round(
                                    dist, 2), np.round(ans, 2), np.round(interp_fn_Bragg(ans), 2))
        self.df_detected = pd.DataFrame(dict(slope=self.detected_slope_list, intercept=self.detected_intercept_list, min_angle=self.detected_line_min_angle_list,
                                             max_angle=self.detected_line_max_angle_list, dE=self.dE_list, gid=self.detected_id_list, roll_angle=self.detected_line_roll_angle_list, centroid_x=self.detected_centroid_x_list, centroid_y=self.detected_centroid_y_list))

    def hkl_roll_separator(self):
        for gid_item, roll in zip(self.df_detected['gid'], self.df_detected['roll_angle']):
            num = [int(s) for s in re.findall(r'-?\d+', str(gid_item))]
            self.h_list.append(num[0])
            self.k_list.append(num[1])
            self.l_list.append(num[2])
            self.roll_list.append(roll)

    def offset_calc_and_plot(self):
        self.pa, self.phen, gid_list = HXRSS_Bragg_generator(
            (self.h_list, self.k_list, self.l_list, self.roll_list, self.pa_range), self.DTHP, self.dthy, self.DTHR, self.alpha)
        self.dE_mean = np.mean(self.df_detected['dE'])
        self.add_plot()
        for E, x, y in zip(self.df_detected['dE'], self.df_detected['centroid_x'], self.df_detected['centroid_y']):
            self.add_text_to_plot(x, y+50, E)
        self.ui.output.setText('Average Energy Offset: '
                               + str(np.round(self.dE_mean, 1))+' eV')

    def img_processing(self):
        self.processed_image = ndimage.grey_dilation(
            self.processed_image, size=(self.d_kernel, self.d_kernel))
        self.processed_image = ndimage.grey_erosion(
            self.processed_image, size=(self.e_kernel, self.e_kernel))

    def add_text_to_plot(self, x, y, E):
        self.text = pg.TextItem(color='w')
        self.img_corr2d.addItem(self.text)
        self.text.setText(str(np.round(E, 1)))
        self.text.setPos(x, y)
        self.text.setZValue(5)
        self.text.show()

    def plot_correl_scan(self):

        #self.doocs_address_label = "dummy label"
        #self.get_device()
        self.event_counter += 1
        n_shots = int(self.ui.sb_n_shots_max.value())
        if len(self.spec_hist) > n_shots:  # add lag value
            self.spec_hist = self.spec_hist[-n_shots:]
            self.doocs_vals_hist = self.doocs_vals_hist[-n_shots:]

        self.spec_hist.append(self.parent.spectrum_event)
        if self.doocs_address_label == 'event':
            self.doocs_vals_hist.append(self.event_counter)
        elif self.doocs_address_label == 'dummy label':
            self.doocs_vals_hist.append(np.sin(time.time()/10)*7.565432 + 25)
        elif self.parent.ui.combo_hirex.currentText() != "DUMMY":
            if self.doocs_dev is None:
                self.ui.sb_corr_2d_run.setChecked(False)
                self.parent.error_box("Wrong DOOCS channel")
                return
            self.doocs_vals_hist.append(self.doocs_dev.get_value())
        else:
            self.doocs_address_label = 'event',
            self.doocs_vals_hist.append(self.event_counter)

        self.sort_and_bin()
        self.np_doocs = self.doocs_bins
        self.np_phen = self.phen_scan
        self.orig_image = self.spec_binned
        #self.phen_res = self.np_phen[2] - self.np_phen[1]
        #self.angle_res = self.np_doocs[2] - self.np_doocs[1]
        #self.min_pangle = min(self.np_doocs)
        #self.max_pangle = max(self.np_doocs)
        self.add_corr2d_image_item()

    def get_device(self):
        mono_no_text = self.ui.combo_mono.currentText()
        num = re.findall(r'\d+', mono_no_text)[0]
        self.mono_no = int(num)
        if self.mono_no == 1:
            self.doocs_address_label = "XFEL.FEL/UNDULATOR.SASE2/MONOPA.2252.SA2/ANGLE"
            #self.doocs_address_label = "dummy label"
        elif self.mono_no == 2:
            self.doocs_address_label = "XFEL.FEL/UNDULATOR.SASE2/MONOPA.2307.SA2/ANGLE"
        eid = self.doocs_address_label
        self.doocs_dev = Device(eid=eid)
        self.doocs_dev.mi = self.mi

    def sort_and_bin(self):
        try:
            bin_doocs = float(self.ui.sb_corr2d_binning.text())  # bin size
        except ValueError:
            bin_doocs = 0

        try:
            phen_min = self.ui.sb_emin.value()/1000
        except ValueError:
            phen_min = -np.inf

        self.phen_orig = self.parent.x_axis

        try:
            phen_max = self.ui.sb_emax.value()/1000
        except ValueError:
            phen_max = np.inf

        d2, d1 = 0, 0

        if phen_max > phen_min:
            d1 = find_nearest_idx(self.phen_orig/1000, phen_min)
            d2 = find_nearest_idx(self.phen_orig/1000, phen_max)
        # else:
        if d2 <= d1:
            d1 = 0
            d2 = len(self.phen_orig)

        self.phen_scan = self.phen_orig[d1:d2]
        n_phens = len(self.phen_scan)

        #print('self.phen_orig',len(self.phen_orig))
        #print('self.phen',len(self.phen))

        if bin_doocs == 0:
            bin_doocs = 1e10

        try:
            self.n_lag = int(self.ui.sb_corr2d_lag.value())  # lag size
        except ValueError:
            self.n_lag = 0

        if len(self.doocs_vals_hist) > abs(self.n_lag)+5:
            if self.n_lag >= 0:
                self.doocs_vals_hist_lagged = self.doocs_vals_hist[:len(
                    self.doocs_vals_hist)-self.n_lag]
                spec_lagged = np.array(self.spec_hist)[self.n_lag:, :]
            else:
                self.doocs_vals_hist_lagged = self.doocs_vals_hist[abs(
                    self.n_lag):]
                spec_lagged = np.array(self.spec_hist)[:len(
                    self.doocs_vals_hist)-abs(self.n_lag), :]

        else:
            self.doocs_vals_hist_lagged = self.doocs_vals_hist
            #spec_lagged = np.array(self.spec_hist) * np.array(self.cross_callibration_vals_hist)[
            #                       :, None] / np.array(self.transmission_vals_hist)[:, None]  # TODO: untested!!!
            spec_lagged = np.array(self.spec_hist)

        min_val = bin_doocs * \
            (int(min(self.doocs_vals_hist_lagged) / bin_doocs))
        max_val = max(self.doocs_vals_hist_lagged) + bin_doocs * 1.01

        if max_val - min_val <= bin_doocs:
            # ensures there is at least one bin (two bin values)
            max_val = min_val + 1.01 * bin_doocs

        self.doocs_bins = np.arange(min_val, max_val, bin_doocs)
        # print('shape of created doocs_bins', self.doocs_bins.shape)

        self.doocs_event_counts, _ = np.histogram(
            self.doocs_vals_hist_lagged, bins=self.doocs_bins)

        self.bin_dest_idx = np.digitize(
            self.doocs_vals_hist_lagged, self.doocs_bins)-1
        self.spec_binned = np.zeros((len(self.doocs_bins)-1, n_phens))

        for i in np.unique(self.bin_dest_idx):
            idx = np.where(i == self.bin_dest_idx)[0]
            # print('sorting', i, idx)
            if len(idx) > 1:
                self.spec_binned[i, :] = np.mean(
                    spec_lagged[idx, d1:d2], axis=0)
                #print('multiple', i, idx, len(self.spec_binned))
            elif len(idx) == 1:
                self.spec_binned[i, :] = spec_lagged[idx[0], d1:d2]
                # else:
                # self.spec_binned = np.array([spec_lagged[idx[0], :]])
            else:
                pass

    def loadStyleSheet(self, filename):
        """
        Sets the dark GUI theme from a css file.
        :return:
        """
        try:
            self.cssfile = "gui/" + filename
            with open(self.cssfile, "r") as f:
                self.setStyleSheet(f.read())
        except IOError:
            logger.error('No style sheet found!')

    def error_box(self, message):
        QtGui.QMessageBox.about(self, "Error box", message)

    def question_box(self, message):
        #QtGui.QMessageBox.question(self, "Question box", message)
        reply = QtGui.QMessageBox.question(self, "Question Box",
                                           message,
                                           QtGui.QMessageBox.Yes | QtGui.QMessageBox.No)
        if reply == QtGui.QMessageBox.Yes:
            return True

        return False

    def open_file(self):  # self.parent.data_dir
        #self.pathname, _ = QtGui.QFileDialog.getOpenFileName(
        #    self, "Open Correlation Data", self.data_dir, 'txt (*.npz)', None, QtGui.QFileDialog.DontUseNativeDialog)
        self.pathname, _ = QtGui.QFileDialog.getOpenFileName(
            self, "Open Correlation Data", "/Users/christiangrech/Nextcloud/Notebooks/HXRSS/Data/npz", 'txt (*.npz)', None, QtGui.QFileDialog.DontUseNativeDialog)
        if self.pathname != "":
            filename = os.path.basename(self.pathname)
            self.ui.file_name.setText(filename)
            self.load_corr2d()
        else:
            self.ui.file_name.setText('')
            self.ui.mono_no.setText('')

    def get_latest_npz(self):
        # * means all if need specific format then *.csv
        #list_of_files = glob.glob(
        #    self.data_dir + "*_cor2d.npz")
        list_of_files = glob.glob(
            '/Users/christiangrech/Nextcloud/Notebooks/HXRSS/Data/npz/' + "*_cor2d.npz")
        #self.pathname = max(list_of_files, key=os.path.getmtime)
        self.pathname = max(list_of_files, key=os.path.getctime)
        self.ui.file_name.setText(os.path.basename(self.pathname))
        print(self.pathname)
        self.load_corr2d()

    def load_corr2d(self):
        self.tt = np.load(self.pathname)
        self.orig_image = self.tt['corr2d']
        self.np_doocs = self.tt['doocs_scale']
        self.np_phen = self.tt['phen_scale']
        self.doocs_label = self.tt['doocs_channel']
        self.info_mono_no()

    def info_mono_no(self):
        if "XFEL.FEL/UNDULATOR.SASE2/MONOPA.2252.SA2/ANGLE" in self.doocs_label or "XFEL.FEL/UNDULATOR.SASE2/MONOPA.2307.SA2/ANGLE" in self.doocs_label:
            if "XFEL.FEL/UNDULATOR.SASE2/MONOPA.2252.SA2/ANGLE" in self.doocs_label:
                self.mono_no = 1
                self.ui.mono_no.setText('Monochromator 1 image found.')
            elif "XFEL.FEL/UNDULATOR.SASE2/MONOPA.2307.SA2/ANGLE" in self.doocs_label:
                self.mono_no = 2
                self.ui.mono_no.setText('Monochromator 2 image found.')
        else:
            self.ui.mono_no.setText('Invalid input')


def main():

    #make pyqt threadsafe
    QtCore.QCoreApplication.setAttribute(QtCore.Qt.AA_X11InitThreads)
    #create the application
    app = QApplication(sys.argv)

    window = UICalculator()

    #show app
    #window.setWindowIcon(QtGui.QIcon('gui/angry_manul.png'))
    # setting the path variable for icon
    path = os.path.join(os.path.dirname(
        sys.modules[__name__].__file__), 'gui/hirex.png')
    app.setWindowIcon(QtGui.QIcon(path))
    window.show()
    window.raise_()
    #Build documentaiton if source files have changed
    #os.system("cd ./docs && xterm -T 'Ocelot Doc Builder' -e 'bash checkDocBuild.sh' &")
    #exit script
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
