"""
Christian Grech, DESY, 2021
based on logger.py by Sergey Tomin
"""
import sys
import numpy as np
import json
from PyQt5 import QtGui, QtCore
from PyQt5.QtWidgets import QWidget, QMessageBox, QApplication
import pyqtgraph as pg
from gui.UICalculator import Ui_Form
import time
import os
from threading import Thread, Event
import logging
from matplotlib import cm
import numpy as np
import pandas as pd
from scipy import ndimage
from scipy.spatial import distance
from scipy.optimize import newton, fsolve
from scipy import interpolate, optimize, stats
from numba import jit
import re
from skimage.transform import hough_line, hough_line_peaks
from model_functions.HXRSS_Bragg_fun_lim import HXRSSsim
from model_functions.HXRSS_Bragg_fun_explorator import HXRSSopt
from itertools import cycle
# filename="logs/afb.log",
#logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Statistics(Thread):
    def __init__(self):
        super(Statistics, self).__init__()
        self._stop_event = Event()
        self.do = None
        self.delay = 0.1

    def run(self):
        while 1:
            self.do()
            logger.info("do")
            time.sleep(self.delay)

    def stop(self):
        self._stop_event.set()


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
        self.linecolors = cycle(self.colors)
        self.linecolors1 = cycle(self.colors)
        self.linecolors2 = cycle(self.colors)
        self.thresh = 0.25
        self.n = 0
        self.d_kernel = 2
        self.e_kernel = 0
        self.mono_no = None
        self.max_E = 1000
        self.max_P = 2
        self.slope_allowance = 3
        self.intercept_allowance = 180
        self.max_distance = 2800
        self.hmax, self.kmax, self.lmax = 5, 5, 5
        self.img_corr2d = None
        self.min_phen = 0
        self.max_phen = 0
        self.min_pangle = 0
        self.max_pangle = 0
        self.dE_mean = 0
        self.pitch_angle_range, self.min_angle_list, self.spec_data_list, self.slope_list, self.y_intercept_list, self.centroid_pa_list, self.centroid_phen_list, self.max_angle_list = [], [], [], [], [], [], [], []
        self.tngnt_slope_list, self.tngnt_intercept_list, self.tngnt_gid_list, self.tngnt_centroid_list, self.tngnt_centroid_y_list, self.tngnt_roll_angle_list, self.interp_Bragg_list = [], [], [], [], [], [], []
        self.detected_slope_list, self.detected_intercept_list, self.detected_id_list, self.detected_line_min_angle_list, self.detected_line_max_angle_list,  self.detected_line_roll_angle_list, self.dE_list, self.dP_list, self.ans_list = [], [], [], [], [], [], [], [], []
        self.h_list, self.k_list, self.l_list, self.roll_list = [], [], [], []
        self.ui.pb_start_calc.clicked.connect(self.start_stop_calc)
        self.ui.browse_button.clicked.connect(self.open_file)
        self.ui.file_name.setText('')
        self.ui.roll_angle.setDecimals(4)
        self.ui.roll_angle.setSuffix(" °")
        self.ui.roll_angle.setRange(0, 2)
        self.ui.roll_angle.setSingleStep(0.001)
        #self.ui = self.parent.ui

    def reset(self):
        self.img_corr2d.clear()
        self.ui.file_name.clear()
        self.ui.mono_no.clear()
        self.ui.roll_angle.clear()

    def save_state(self, filename):
        table = {}
        #table["sr_period"] = self.sb_sr_period.value()

        #with open(filename, 'w') as f:
        #    json.dump(table, f)
        #logger.info("Save State")
        pass

    def load_state(self, filename):
        #with open(filename, 'r') as f:
        #    table = json.load(f)

        #if "sr_delay" in table: self.sb_sr_delay.setValue(table["sr_delay"])
        #logger.info("Load State")
        pass

    def closeEvent(self, QCloseEvent):
        self.stop_calc()
        self.reset()

    def start_stop_calc(self):
        if self.ui.pb_start_calc.text() == "Stop":
            self.stop_calc()
            self.reset()
        else:
            if self.ui.mono_no.text() == "Invalid input":
                self.error_box("Select a valid npz file first")
                return
            if self.ui.mono_no.text() == "":
                self.error_box("Select a valid npz file first")
                return
            self.binarization()
            self.get_binarized_line()
            self.img_processing()
            self.get_binarized_line()
            self.add_corr2d_image_widget()
            #self.hough_line_transform()
            #self.generate_Bragg_curves()
            #self.tangent_generator()
            #self.line_comparator()
            self.add_plot()
            #if len(self.df_detected.index) != 0:
            #    self.hkl_roll_separator()
            # Get Bragg curves
            #    pa, phen, linestyle_list, gid_list = HXRSSopt(
            #        (self.h_list, self.k_list, self.l_list, self.roll_list, self.pa_range), self.DTHP, self.dthy, self.DTHR, self.alpha)

            #for r in range(len(pa)):
            #    ax[1].plot(pa[r], phen[r], color=next(linecolors2))

            #    self.dE_mean = np.mean(self.dE_list)

            #    self.ui.mono_no.setText('Energy offset: '
            #+ str(round(self.dE_mean, 1))+' eV')
            # If no lines are detected
            #else:
            #    logger.info('No lines can be matched')

            self.ui.pb_start_calc.setText("Reset")

            self.ui.pb_start_calc.setStyleSheet(
                "color: rgb(63, 191, 95); font-size: 18pt")

    def add_corr2d_image_widget(self):
        win = pg.GraphicsLayoutWidget()
        layout = QtGui.QGridLayout()
        self.ui.widget_calc.setLayout(layout)
        layout.addWidget(win)

        self.img_corr2d = win.addPlot()
        self.add_corr2d_image_item()

    def add_corr2d_image_item(self):
        self.img_corr2d.clear()

        self.img_corr2d.setLabel('left', "E_ph", units='eV')
        self.img_corr2d.setLabel('bottom', "Pitch angle", units='°')

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

    def binarization(self):
        # all values below 0 threshold are set to 0
        self.corr2d[self.corr2d < 0] = 0
        # define parameters for binarization
        range_scale = np.ptp(self.corr2d)
        threshold = self.thresh * range_scale
        max_value = np.amax(self.corr2d)
        min_value = np.amin(self.corr2d)
        # all values above threshold are set to max_value
        self.corr2d[self.corr2d > threshold] = max_value
        # all values above threshold are set to min_value
        self.corr2d[self.corr2d < threshold] = min_value
        self.processed_image = self.corr2d.T

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
        for _, pitch_angle, rho in zip(*hough_line_peaks(h, theta, d, num_peaks=5, min_distance=10, min_angle=10)):

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
            pen = pg.mkPen(next(self.linecolors), width=5,
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
            self.DTHP = -0.382
            self.dthy = 1.17
            self.DTHR = 0.04
            self.alpha = 0.0028
        else:
            self.DTHP = -0.382
            self.dthy = 1.17
            self.DTHR = 0.04
            self.alpha = 0.0028
        self.pa_range = np.linspace(self.min_pangle-1, self.max_pangle+1, 200)
        # pass pitch and roll errors and create Bragg curves
        self.phen_list, self.p_angle_list, self.gid_list, self.roll_angle_list = HXRSSsim(
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
                if y1[0] < max(self.np_phen)+1000 and y1[0] > min(self.np_phen)-1000:
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
            pitch_angle_range = np.linspace(min_angle, max_angle, 10)
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
                            self.dE_list.pop()
                            self.dP_list.pop()
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
                            self.dP_list.append(dP)
                            n = n+1
                            logger.info('Its a match ', n, ' Curve id:', curve_id, 'Distance', np.round(
                                    dist, 2), np.round(ans, 2), np.round(interp_fn_Bragg(ans), 2))
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
                            self.dP_list.append(dP)
                            n = n+1
                            print('Its a match ', n, ' Curve id:', curve_id, 'Distance', np.round(
                                    dist, 2), np.round(ans, 2), np.round(interp_fn_Bragg(ans), 2))
        self.df_detected = pd.DataFrame(dict(slope=self.detected_slope_list, intercept=self.detected_intercept_list, min_angle=self.detected_line_min_angle_list, max_angle=self.detected_line_max_angle_list, dE=self.dE_list,
                                             dP=self.dP_list, gid=self.detected_id_list, roll_angle=self.detected_line_roll_angle_list))

    def hkl_roll_separator(self):
        for gid_item, roll in zip(self.df_detected['gid'], self.df_detected['roll_angle']):
            num = [int(s) for s in re.findall(r'-?\d+', str(gid_item))]
            self.h_list.append(num[0])
            self.k_list.append(num[1])
            self.l_list.append(num[2])
            self.roll_list.append(roll)

    def add_plot(self):
        gui_index = self.parent.ui.get_style_name_index()
        if "standard" in self.parent.gui_styles[gui_index]:
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

        self.plot1.setLabel('left', "E_ph", units='eV')
        self.plot1.setLabel('bottom', "Pitch angle", units='deg')

        self.plot1.showGrid(1, 1, 1)

        self.plot1.getAxis('left').enableAutoSIPrefix(
            enable=False)  # stop the auto unit scaling on y axes
        layout = QtGui.QGridLayout()
        self.ui.widget_calc_2.setLayout(layout)
        layout.addWidget(win, 0, 0)

        self.plot1.setAutoVisible(y=True)

        self.plot1.addLegend()

        self.single = pg.PlotCurveItem(pen=single_pen, name='single')

        self.plot1.addItem(self.single)

        pen = pg.mkPen((51, 255, 51), width=2)
        pen = pg.mkPen((255, 0, 0), width=3)
        #self.average = pg.PlotCurveItem(x=[], y=[], pen=pen, name='average')
        self.average = pg.PlotCurveItem(pen=pen, name='average')

        self.plot1.addItem(self.average)

        pen = pg.mkPen((0, 255, 255), width=2)

        #self.plot1.addItem(self.back_plot) ##################################### SS removed, as typically we don;t need it once start pySpectrometer

        # cross hair
        self.vLine = pg.InfiniteLine(angle=90, movable=False)
        self.hLine = pg.InfiniteLine(angle=0, movable=False)
        self.plot1.addItem(self.vLine, ignoreBounds=True)
        self.plot1.addItem(self.hLine, ignoreBounds=True)

    def stop_calc(self):
        logger.info("Stop Logger")
        self.ui.pb_start_calc.setStyleSheet(
            "color: rgb(255, 0, 0); font-size: 18pt")
        self.ui.pb_start_calc.setText("Start")

    def img_processing(self):
        self.processed_image = ndimage.grey_dilation(
            self.processed_image, size=(self.d_kernel, self.d_kernel))
        self.processed_image = ndimage.grey_erosion(
            self.processed_image, size=(self.e_kernel, self.e_kernel))

    def zoom_signal(self):
        pass
        #s_up = self.plot_y.viewRange()[0][0]
        #s_down = self.plot_y.viewRange()[0][1]

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

    def open_file(self):
        pathname = QtGui.QFileDialog.getOpenFileName(
            self, 'Open Correlation Data', '(*.npz)')
        filename = os.path.basename(pathname[0])
        self.ui.file_name.setText(filename)
        tt = np.load(pathname[0])
        self.corr2d = tt['corr2d']
        self.orig_image = tt['corr2d']
        self.np_doocs = tt['doocs_scale']
        self.np_phen = tt['phen_scale']
        self.doocs_label = tt['doocs_channel']
        self.phen_res = self.np_phen[2] - self.np_phen[1]
        self.angle_res = self.np_doocs[2] - self.np_doocs[1]
        self.min_pangle = min(self.np_doocs)
        self.max_pangle = max(self.np_doocs)
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