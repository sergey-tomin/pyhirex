"""
Christian Grech, DESY, 2021
based on logger.py by Sergey Tomin
"""
import sys
import numpy as np
import pathlib
from PyQt5 import QtGui, QtCore, QtWidgets
from PyQt5.QtGui import QBrush, QColor
from PyQt5.QtWidgets import QWidget, QApplication
import pyqtgraph as pg
from gui.UICalculator import Ui_Form
import os
import glob
import logging
from mint.xfel_interface import *
# import pydoocs
from matplotlib import cm
import pandas as pd
from scipy import ndimage
from datetime import datetime, timedelta
from skimage.filters import threshold_yen
#from sklearn.model_selection import cross_val_score
from sklearn.neighbors import NearestNeighbors, KNeighborsClassifier
#from sklearn.ensemble import GradientBoostingClassifier, AdaBoostClassifier, RandomForestClassifier, VotingClassifier
#from sklearn.linear_model import LogisticRegression
#from sklearn.tree import DecisionTreeClassifier
from gui.spectr_gui import send_to_desy_elog
from sklearn import preprocessing
from scipy import interpolate
import re
from skimage.transform import hough_line, hough_line_peaks
from model_functions.HXRSS_Bragg_max_generator import HXRSS_Bragg_max_generator
from model_functions.HXRSS_Bragg_single import HXRSSsingle


path = os.path.realpath(__file__)
indx = path.find("hirex.py")
print("PATH to main file: " + os.path.realpath(__file__)
      + " path to folder: " + path[:indx])
sys.path.insert(0, path[:indx])
# filename="logs/afb.log",
#logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
pd.options.mode.chained_assignment = None  # default='warn'

PY_SPECTROMETER_DIR = "pySpectrometer"
DIR_NAME = "hirex"


class UICalculator(QWidget):
    def __init__(self, parent=None):
        super().__init__()
        self.parent = parent
        self.ui = Ui_Form()
        self.ui.setupUi(self)
        gui_index = self.parent.ui.get_style_name_index()
        style_name = self.parent.gui_styles[gui_index]
        self.loadStyleSheet(filename=self.parent.gui_dir + style_name)
        self.mi = self.parent.mi
        # Initialize flags
        self.nomatch = 0
        self.allow_data_storage = 0
        # Initialize parameters
        self.mode = 0
        self.mono_no = None
        self.min_phen, self.max_phen = 0, 0
        self.min_pangle, self.max_pangle = 0, 0
        self.img_corr2d = None
        self.dE_mean = 0
        self.pixel_calibration_mean = 0
        self.yvalue = []
        self.pitch_angle_range, self.min_angle_list, self.spec_data_list, self.slope_list, self.y_intercept_list, self.centroid_pa_list, self.centroid_phen_list, self.max_angle_list = [], [], [], [], [], [], [], []
        self.tngnt_slope_list, self.tngnt_intercept_list, self.tngnt_gid_list, self.tngnt_centroid_list, self.tngnt_centroid_y_list, self.tngnt_roll_angle_list, self.interp_Bragg_list = [], [], [], [], [], [], []
        self.detected_slope_list, self.detected_intercept_list, self.detected_id_list, self.detected_line_min_angle_list, self.detected_line_max_angle_list,  self.detected_line_roll_angle_list, self.actual_E, self.dE_list, self.ans_list, self.detected_centroid_x_list, self.detected_centroid_y_list = [], [], [], [], [], [], [], [], [], [], []
        self.h_list, self.k_list, self.l_list, self.roll_list, self.centroid_list = [], [], [], [], []
        self.ind = ''

        # Set folder directory path to save and obtain files from SASE2 folder
        DIR_NAME = os.path.basename(pathlib.Path(__file__).parent.absolute())
        self.path = path[:path.find(DIR_NAME)]
        self.data_dir = path[:path.find(
            "user")] + "user" + os.sep + PY_SPECTROMETER_DIR + os.sep + "SASE2" + os.sep

        # Connect UI buttons and text displays
        self.ui.pb_start_calc.clicked.connect(self.start_stop_calc_from_npz)
        self.ui.browse_button.clicked.connect(self.open_file)
        self.ui.pb_logbook.clicked.connect(
            lambda: self.logbook(self.ui.tab, text="Suggested energy shift by "+str(np.round(self.dE_mean, 1))+" eV and a pixel calibration of " + str(self.pixel_calibration_mean)))
        self.ui.file_name.setText('')
        self.ui.roll_angle.setDecimals(4)
        self.ui.roll_angle.setSuffix(" °")
        self.ui.roll_angle.setRange(0, 2)
        self.ui.roll_angle.setValue(1.5013)
        self.ui.roll_angle.setSingleStep(0.001)
        self.ui.tableWidget.setRowCount(0)
        # Check if scan is recent and if yes allow DOOCS push
        self.ui.pb_doocs.clicked.connect(self.check_if_scan_is_recent)
        self.ui.pb_load_doocs.clicked.connect(self.load_from_doocs)
        # Set constants
        self.hmax, self.kmax, self.lmax = 6, 6, 7
        self.d_kernel, self.e_kernel = 2, 2
        # Set up and show the two graph axes and display latest npz file
        self.add_image_widget()
        self.add_plot_widget()
        self.get_latest_npz()

    def reset(self):
        self.dE_mean, self.min_phen, self.max_phen = 0, 0, 0
        self.min_pangle, self.max_pangle = 0, 0
        self.dE_mean = 0
        self.pixel_calibration_mean = 0
        self.ind = ''
        self.pitch_angle_range, self.min_angle_list, self.spec_data_list, self.slope_list, self.y_intercept_list, self.centroid_pa_list, self.centroid_phen_list, self.max_angle_list = [], [], [], [], [], [], [], []
        self.tngnt_slope_list, self.tngnt_intercept_list, self.tngnt_gid_list, self.tngnt_centroid_list, self.tngnt_centroid_y_list, self.tngnt_roll_angle_list, self.interp_Bragg_list = [], [], [], [], [], [], []
        self.detected_slope_list, self.detected_intercept_list, self.detected_id_list, self.detected_line_min_angle_list, self.detected_line_max_angle_list,  self.detected_line_roll_angle_list, self.dE_list, self.ans_list, self.detected_centroid_x_list, self.detected_centroid_y_list, self.actual_E = [], [], [], [], [], [], [], [], [], [], []
        self.h_list, self.k_list, self.l_list, self.roll_list, self.roll_list_fun, self.pa, self.phen, self.gid_list, self.centroid_list = [
            ], [], [], [], [], [], [], [], []
        self.ui.tableWidget.setRowCount(0)
        self.ui.pb_doocs.setEnabled(False)
        self.ui.pb_logbook.setEnabled(False)
        if self.mode == 1:
            self.img_corr2d.clear()
            self.plot1.clear()
            if self.nomatch == 0:
                self.legend.scene().removeItem(self.legend)
                self.model.setData(x=[], y=[])
            self.ui.output.setText('')
            self.info_mono_no()
            self.ui.pb_start_calc.setStyleSheet(
                "color: rgb(85, 255, 127); font-size: 14pt")
            self.ui.pb_start_calc.setText("Calculate fom npz file")
            self.mode = 0

    def closeEvent(self, QCloseEvent):
    	self.mode = 0
    	self.reset()

    def start_stop_calc_from_npz(self):
        self.mode = 1
        if self.ui.pb_start_calc.text() == "Reset":
            self.reset()
        else:
            if self.ui.status.text() == "Invalid input\n":
                self.error_box("Select a valid npz file first")
                return
            if self.ui.status.text() == "":
                self.error_box("Select a valid npz file first")
                return
            self.load_corr2d()
            self.corr2d = self.tt['corr2d']
            if len(self.np_doocs) > 2:
                self.nomatch = 0
                #self.angle_res = self.np_doocs[2] - self.np_doocs[1]
                self.scale_xaxis = (max(self.np_doocs)
                                    - min(self.np_doocs)) / len(self.np_doocs)
                self.angle_res = self.scale_xaxis
            else:
                self.nomatch = 1
                self.ui.output.setText(
                    self.ui.output.text() + 'Pitch angle range too small\n')
                self.ui.pb_start_calc.setText("Reset")
                self.ui.pb_start_calc.setStyleSheet(
                                "color: rgb(255, 0, 0); font-size: 14pt")
                return
            self.binarization()
            self.ui.output.setText(
                self.ui.output.text() + 'Image binarization complete\n')
            self.get_binarized_line()
            self.img_processing()
            self.add_corr2d_image_item()
            self.hough_line_transform()
            self.generate_Bragg_curves()
            if len(self.df_spec_lines.index) != 0:
                self.tangent_generator()
                #self.line_comparator()
                self.nearest_neighbor()
                if len(self.df_detected.index) != 0:
                    self.hkl_roll_separator()
                # Get Bragg curves
                    self.offset_calc_and_plot()
                # If no lines are detected
                else:
                    self.ui.output.setText(
                        self.ui.output.text() + 'No lines can be matched\n')
                    self.nomatch_plot()
            else:
                self.ui.output.setText(
                        self.ui.output.text() + 'No lines were detected in image\n')
                self.nomatch_plot()
            self.ui.pb_start_calc.setText("Reset")
            self.ui.pb_start_calc.setStyleSheet(
                "color: rgb(255, 0, 0); font-size: 14pt")

    def add_image_widget(self):
        self.win1 = pg.GraphicsLayoutWidget()
        self.layout = QtGui.QGridLayout()
        self.ui.widget_calc.setLayout(self.layout)
        self.layout.addWidget(self.win1)
        self.img_corr2d = self.win1.addPlot()
        self.img_corr2d.setLabel('left', "E_HIREX", units='eV')
        self.img_corr2d.setLabel('bottom', "Pitch angle", units='°')
        self.img_corr2d.getAxis('left').enableAutoSIPrefix(
                    enable=False)  # stop the auto unit scaling on y axes

    def add_corr2d_image_item(self):
        self.img_corr2d.clear()
        self.scale_yaxis = (
            self.np_phen[-1] - self.np_phen[0]) / len(self.np_phen)
        translate_yaxis = self.np_phen[0] / self.scale_yaxis
        translate_xaxis = min(self.np_doocs) / self.scale_xaxis

        self.img = pg.ImageItem()
        self.img_corr2d.addItem(self.img)
        colormap = cm.get_cmap('viridis')
        colormap._init()
        # Convert matplotlib colormap from 0-1 to 0 -255 for Qt
        lut = (colormap._lut * 255).view(np.ndarray)
        # Apply the colormap
        self.img.setLookupTable(lut)
        self.img.setImage(self.orig_image)
        self.img.scale(self.scale_xaxis, self.scale_yaxis)
        self.img.translate(translate_xaxis, translate_yaxis)

    def add_plot_widget(self):
        gui_index = self.parent.ui.get_style_name_index()
        if "standard" in self.parent.gui_styles[gui_index]:
            pg.setConfigOption('background', 'w')
            pg.setConfigOption('foreground', 'k')
            model_pen = pg.mkPen("k")
        else:
            model_pen = pg.mkPen("w")

        self.win2 = pg.GraphicsLayoutWidget()
        self.label = pg.LabelItem(justify='left', row=0, col=0)
        self.win2.addItem(self.label)
        self.vb = self.win2.addViewBox(row=1, col=1)
        self.vb.setMaximumWidth(100)
        self.plot1 = self.win2.addPlot(row=1, col=0)
        self.plot1.setLabel('left', "E_ph", units='eV')
        self.plot1.setLabel('bottom', "Pitch angle", units='°')
        self.plot1.showGrid(1, 1, 1)
        self.plot1.getAxis('left').enableAutoSIPrefix(
            enable=False)  # stop the auto unit scaling on y axes
        self.layout_2 = QtGui.QGridLayout()
        self.ui.widget_calc_2.setLayout(self.layout_2)
        self.layout_2.addWidget(self.win2, 0, 0)
        self.plot1.setAutoVisible(y=True)

        # cross hair
        self.vLine = pg.InfiniteLine(angle=90, movable=False)
        self.hLine = pg.InfiniteLine(angle=0, movable=False)
        self.plot1.addItem(self.vLine, ignoreBounds=True)
        self.plot1.addItem(self.hLine, ignoreBounds=True)
        self.plot1.setXLink(self.img_corr2d)
        #self.plot1.setYLink(self.img_corr2d)

    def add_plot(self):
        #self.plot1.clear()
        self.plot1.enableAutoRange()
        #pen_shifted = pg.mkPen('k', width=3, style=QtCore.Qt.DashLine)
        self.legend = self.plot1.addLegend()
        self.legend.setParentItem(self.vb)

    # Anchor the upper-left corner of the legend to the upper-left corner of its parent
        self.legend.anchor((0, 0), (0, 0))
        #self.legend_boolean = 1
        for r in range(len(self.pa)):
            if self.linestyle_list[r] == 'dashed':
                style_type = QtCore.Qt.DashLine
            if self.linestyle_list[r] == 'solid':
                style_type = QtCore.Qt.SolidLine
            if self.linestyle_list[r] == 'dashdot':
                style_type = QtCore.Qt.DashDotLine
            pen = pg.mkPen(str(self.color_list[r]), width=3, style=style_type)
            self.model = pg.PlotCurveItem(
                x=self.pa[r], y=self.phen[r], pen=pen, name=self.gid_list[r])
            if self.phen[r][50] <= max(self.np_phen)+1500 and self.phen[r][50] >= min(self.np_phen)-1500:
                self.plot1.addItem(self.model)
        self.plot1.setXRange(min(self.np_doocs),
                             max(self.np_doocs), padding=None, update=True)

    def add_table_row(self, col1, col2, col3):
        rowPosition = self.ui.tableWidget.rowCount()
        self.ui.tableWidget.insertRow(rowPosition)  # insert new row
        item1 = QtGui.QTableWidgetItem(col1)
        item2 = QtGui.QTableWidgetItem(col2)
        item3 = QtGui.QTableWidgetItem(col3)
        self.ui.tableWidget.setItem(
            rowPosition, 0, item1)
        self.ui.tableWidget.setItem(
            rowPosition, 1, item2)
        self.ui.tableWidget.setItem(
            rowPosition, 2, item3)
        if self.ind == 'error':
            item1.setForeground(QBrush(QColor(255, 0, 0)))
            item2.setForeground(QBrush(QColor(255, 0, 0)))
            item3.setForeground(QBrush(QColor(255, 0, 0)))
        if self.ind == 'record':
            item1.setForeground(QBrush(QColor(0, 0, 255)))
            item2.setForeground(QBrush(QColor(0, 0, 255)))
            item3.setForeground(QBrush(QColor(0, 0, 255)))
        self.ind = ''

    def binarization(self):
        # all values below 0 threshold are set to 0
        self.phen_res = self.np_phen[2] - self.np_phen[1]
        self.min_pangle = min(self.np_doocs)
        self.max_pangle = max(self.np_doocs)
        self.corr2d[self.corr2d < 0] = 0
        self.image = self.corr2d.T
        thresh = threshold_yen(self.image, nbins=256)
        binary = self.image > thresh
        self.processed_image = binary
        #### ALTERNATE MANUAL THRESHOLDING
        #range_scale = np.ptp(self.corr2d)
        #threshold = 0.16 * range_scale
        #max_value = np.amax(self.corr2d)
        #min_value = np.amin(self.corr2d)
        # all values above threshold are set to max_value
        #self.corr2d[self.corr2d > threshold] = max_value
        # all values above threshold are set to min_value
        #self.corr2d[self.corr2d < threshold] = min_value
        #self.processed_image = self.corr2d.T

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
            h, theta, d, num_peaks=5, min_distance=20, min_angle=20)
        if len(pitch_angle_list) == 0:
            self.ui.output.setText(
                self.ui.output.text() + 'No lines detected\n')
        else:
            self.ui.output.setText(self.ui.output.text(
            ) + '%d line(s) found\n' % len(pitch_angle_list))
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
                self.ui.output.setText(
                                self.ui.output.text() + 'Horizontal line ignored\n')
                continue
            if np.isneginf(slope) or np.isposinf(slope):
                self.ui.output.setText(
                            self.ui.output.text() + 'Vertical line ignored\n')
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
        self.df_spec_lines['roll_angle'] = self.set_roll_angle

    def generate_Bragg_curves(self):
        self.roll = list(self.df_spec_lines['roll_angle'])
        if self.mono_no == 2:
            self.DTHP = -0.392
            self.dthy = 1.17
            self.DTHR = -0.1675
            self.alpha = 0.00238
        else:
            self.DTHP = -0.392
            self.dthy = 1.17
            self.DTHR = -0.1675
            self.alpha = 0.00238
        self.pa_range = np.linspace(self.min_pangle-1, self.max_pangle+1, 100)
        self.pa_range_plot = np.linspace(
            self.min_pangle-1, self.max_pangle+1, 100)
        # pass pitch and roll errors and create Bragg curves
        self.phen_list, self.p_angle_list, self.gid_list, self.roll_angle_list, color_list, linestyle_list = HXRSS_Bragg_max_generator(
            self.pa_range, self.hmax, self.kmax, self.lmax, self.DTHP, self.dthy, self.roll, self.DTHR, self.alpha)

    def tangent_generator(self):
        for r, gid_raw, roll_angle in zip(range(len(self.p_angle_list)), self.gid_list, self.roll_angle_list):
            x = np.asarray(self.p_angle_list[r])
            y = np.asarray(self.phen_list[r])
            # Interpolating range
            x0 = np.linspace(min(self.p_angle_list[r]), max(
                self.p_angle_list[r]), 150, endpoint=False)

            gid = str(gid_raw)
            f = interpolate.UnivariateSpline(
                self.p_angle_list[r], self.phen_list[r])
            for x0_ in x0:
                i0 = np.argmin(np.abs(x-x0_))
                x1 = x[i0:i0+2]
                y1 = y[i0:i0+2]
                dydx, = np.diff(y1)/np.diff(x1)
                if y1[0] < max(self.np_phen)+250 and y1[0] > min(self.np_phen)-250:
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

    def nearest_neighbor(self):
        scaler = preprocessing.MinMaxScaler()
        self.df_tangents_scaled = pd.DataFrame(scaler.fit_transform(self.df_tangents[['slope', 'intercept', 'centroid_pa', 'centroid_phen']]), columns=self.df_tangents[[
                                               'slope', 'intercept', 'centroid_pa', 'centroid_phen']].columns)
        self.df_test = self.df_spec_lines[[
            'slope', 'intercept', 'centroid_pa', 'centroid_phen', 'min_angle', 'max_angle', 'roll_angle']]
        self.df_test_scaled = pd.DataFrame(scaler.transform(self.df_test[['slope', 'intercept', 'centroid_pa', 'centroid_phen']]), columns=self.df_test[[
                                           'slope', 'intercept', 'centroid_pa', 'centroid_phen']].columns)
        X = self.df_tangents_scaled
        y = self.df_tangents['gid']
        #clf = RandomForestClassifier(n_estimators=5, random_state=1)
        clf = KNeighborsClassifier(
            n_neighbors=13, weights='distance', algorithm='auto')
        clf.fit(X, y)
        self.df_test['gid'] = clf.predict(self.df_test_scaled)
        self.df_detected = pd.DataFrame(dict(slope=self.df_test['slope'], intercept=self.df_test['intercept'], min_angle=self.df_test['min_angle'], max_angle=self.df_test['max_angle'],
                                             gid=self.df_test['gid'], roll_angle=self.df_test['roll_angle'], centroid_pa=self.df_test['centroid_pa'], centroid_phen=self.df_test['centroid_phen']))
        #

    def dispersion_cal(self):
        pixel_calib_list = []
        for slope, mdl_slope, curve_id, centroid_pa in zip(self.df_detected['slope'], self.df_detected['mdl_slope'], self.df_detected['gid'], self.df_detected['centroid_pa']):
            pixel_cal = mdl_slope/slope
            msg = 'Id:' + curve_id + ' matched to line with centroid: ' + \
                str(np.round(centroid_pa, 1)) + ' deg\n'
            self.ui.output.setText(self.ui.output.text() + msg)
            if abs(pixel_cal) > 1.25 or abs(pixel_cal) < 0.75:
                self.ind = 'error'
            else:
                self.add_table_row(curve_id + 'ev/px', str(np.round(self.scale_yaxis, 3)), str(
                    np.round(self.scale_yaxis*pixel_cal, 3)))
                pixel_calib_list.append(self.scale_yaxis*pixel_cal)
        self.pixel_calibration_mean = np.mean(pixel_calib_list)

    def energy_off_cal(self):
        # Subtract model energy and measured energy to get offset dE
        self.df_detected['dE'] = self.df_detected['E_model'] - \
            self.df_detected['centroid_phen']
        print(self.df_detected['E_model'], self.df_detected['centroid_phen'])
        self.actual_E_mean = np.mean(self.df_detected['E_model'])
        # Remove any dE values outside the following range
        btwn = self.df_detected['dE'].between(-290, 290, inclusive=False)
        self.df_detected = self.df_detected[btwn]
        # Print separate row for each detected line and calcuated offset in eV
        for E, id in zip(self.df_detected['dE'], self.df_detected['gid']):
            if abs(E) > 300:
                self.ind = 'error'
            self.add_table_row(
                id + ' Eoff', '-', str(np.round(E, 1))+' eV')
        self.dE_mean = np.mean(self.df_detected['dE'])

    def calculate_means(self):
        if np.isnan(self.dE_mean) is True:
            self.dE_mean = 0
        if np.isnan(self.pixel_calibration_mean) is True:
            self.pixel_calibration_mean = 0
        # Print in red if value is outside range otherwise print in blue
        if abs(self.pixel_calibration_mean) > 1:
            self.ind = 'error'
        else:
            self.ind = 'record'
        self.add_table_row(
            'Avg. ev/px', '-', str(np.round(self.pixel_calibration_mean, 3)))
        self.add_plot()
        self.plot1.setYRange(min(self.np_phen)-100,
                             max(self.np_phen)+100, padding=None, update=True)
        if abs(self.dE_mean) > 300:
            self.ind = 'error'
        self.add_table_row(
            'Avg. Eoff', '-', str(np.round(self.dE_mean, 1))+' eV')
        self.ind = 'record'  # Make sure to list Eo in blue as it will be recorded
        self.add_table_row('HIREX Eo', str(np.round(self.parent.ui.sb_E0.value(), 0)) + ' eV', str(
            np.round((self.parent.ui.sb_E0.value()+self.dE_mean), 0))+' eV')
        #self.add_table_row('Actual E_ph', str(
        #    np.round(self.actual_E_mean, 0))+' eV', '')
        for oldE, E, pa, id in zip(self.df_detected['centroid_phen'], self.df_detected['E_model'], self.df_detected['centroid_pa'], self.df_detected['gid']):
            self.add_table_row(
                ' Eph at ' + str(round(pa, 2)), str(np.round(oldE, 0))+' eV', str(np.round(E, 0))+' eV')
        self.add_table_row(' ', ' ', ' ')

    def hkl_roll_separator(self):
        for gid_item, roll, cent_x in zip(self.df_detected['gid'], self.df_detected['roll_angle'], self.df_detected['centroid_pa']):
            num = [int(s) for s in re.findall(r'-?\d+', str(gid_item))]
            if not(abs(num[0])+abs(num[1])+abs(num[2])==5) and not(abs(num[0])+abs(num[1])+abs(num[2])==13):
                self.h_list.append(num[0])
                self.k_list.append(num[1])
                self.l_list.append(num[2])
                self.roll_list.append(roll)
                self.centroid_list.append(cent_x-self.DTHP)
            else:
                self.ui.output.setText(self.ui.output.text(
                            ) + 'Skipped reflection ' + str(num) + '.\n')

    def offset_calc_and_plot(self):
        self.roll_list_fun = [self.set_roll_angle]
        self.phen, self.pa, gid_list, _roll_list, self.color_list, self.linestyle_list = HXRSS_Bragg_max_generator(
            self.pa_range_plot, self.hmax, self.kmax, self.lmax, self.DTHP, self.dthy, self.roll_list_fun, self.DTHR, self.alpha)

        # Get energy value at one particular pitch angle value, in order to calculate offset
        pa_dE, phen_Actual, gid_list_s, model_slope_list = HXRSSsingle(
            (self.h_list, self.k_list, self.l_list, self.roll_list, self.centroid_list), self.DTHP, self.dthy, self.DTHR, self.alpha)

        self.df_model = pd.DataFrame(
            dict(E_model=phen_Actual, gid=gid_list_s, centroid_pa=pa_dE, mdl_slope=model_slope_list))
        # Merge model phen values with detected lines phen
        self.df_detected = self.df_detected.merge(
            self.df_model, on=['gid', 'centroid_pa'], how='left')
        self.energy_off_cal()
        # Calculate pixel calibration
        self.dispersion_cal()
        # Calculate mean energy offset and pixel calibration
        self.calculate_means()
        # Remove NaN values
        self.allow_data_storage = 1  # File will be created with all parameters calculated
        # Enable logbook button
        self.ui.pb_logbook.setEnabled(True)
        self.ui.pb_doocs.setEnabled(True)

    def nomatch_plot(self):
        self.roll_list = [self.set_roll_angle]
        self.phen, self.pa, self.gid_list, _roll_list, self.color_list, self.linestyle_list = HXRSS_Bragg_max_generator(
            self.pa_range_plot, self.hmax, self.kmax, self.lmax, self.DTHP, self.dthy, self.roll_list, self.DTHR, self.alpha)
        if len(self.pa) > 0:
            self.add_plot()
            self.plot1.setYRange(min(self.np_phen),
                                 max(self.np_phen), padding=None, update=True)
            self.ui.output.setText(self.ui.output.text(
                ) + 'No calibration offset value calculated but possible lines plotted on the right.\n')
        else:
            # In case no lines are plotted, this flag makes sure Legend is not reset (causing an error as there is no legend)
            self.nomatch = 1
            self.ui.output.setText(self.ui.output.text(
                            ) + 'No calibration offset value calculated and no model lines in the area.\n')

    # Dilate and erode pixels in binarized image
    def img_processing(self):
        self.processed_image = ndimage.grey_dilation(
            self.processed_image, size=(self.d_kernel, self.d_kernel))
        self.processed_image = ndimage.grey_erosion(
            self.processed_image, size=(self.e_kernel, self.e_kernel))

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
            self.write_doocs()
            return True

        return False

    def get_screenshot(self, window_widget):
        screenshot_tmp = QtCore.QByteArray()
        screeshot_buffer = QtCore.QBuffer(screenshot_tmp)
        screeshot_buffer.open(QtCore.QIODevice.WriteOnly)
        widget = QtWidgets.QWidget.grab(window_widget)
        widget.save(screeshot_buffer, "png")
        return screenshot_tmp.toBase64().data().decode()

    def logbook(self, widget, text=""):
        """
        Method to send data + screenshot to eLogbook
        :return:
        """
        screenshot = self.get_screenshot(widget)
        device = self.parent.ui.combo_hirex.currentText()
        res = send_to_desy_elog(author="", title="pySpectrometer absolute energy calibration " + device, severity="INFO", text=text, elog=self.mi.logbook_name,
                                image=screenshot)
        if not res:
            self.Form.error_box("error during eLogBook sending")
        if self.allow_data_storage == 1:
            self.save_calc_data_as()

    def save_calc_data_as(self):
        file_timestamp = os.path.splitext(self.ui.file_name.text())[0]
        filename = self.data_dir + file_timestamp + "_en_calib_calc.npz"
        np.savez(filename, dE_mean=self.dE_mean,
                 pix_calib=self.pixel_calibration_mean, details=self.df_detected)
        self.allow_data_storage = 0

    def check_if_scan_is_recent(self):
        self.file_name = os.path.splitext(self.ui.file_name.text())[0]
        file_timestamp_filt = self.file_name[0: 17]
        self.date_time_obj = datetime.strptime(
            file_timestamp_filt, '%Y%m%d-%H_%M_%S')
        present = datetime.now()
        deltat = present - self.date_time_obj
        if deltat < timedelta(days=30):
            self.ui.output.setText(self.ui.output.text(
                            ) + 'Results are recent enough to push to DOOCS. Parameters in blue can be pushed.')
            self.write_doocs()
        else:
            self.question_box(
                "This scan may not be recent enough to update DOOCS parameters. Do you still want to proceed with writing to DOOCS?")

    def write_doocs(self):
        self.doocs_permit = True
        try:
            pydoocs.write(
                "XFEL.UTIL/DYNPROP/HIREX.SA2/PIXEL_CALIBRATION", self.pixel_calibration_mean)
            self.pixel_doocs = pydoocs.read(
                "XFEL.UTIL/DYNPROP/HIREX.SA2/PIXEL_CALIBRATION")
            pydoocs.write(
                "XFEL.UTIL/DYNPROP/HIREX.SA2/CENTRAL_ENERGY", self.dE_mean)
            self.central_doocs = pydoocs.read(
                "XFEL.UTIL/DYNPROP/HIREX.SA2/CENTRAL_ENERGY")
            pydoocs.write(
                "XFEL.UTIL/DYNPROP/HIREX.SA2/FILENAME", self.file_name)
            self.filename_doocs = pydoocs.read(
                "XFEL.UTIL/DYNPROP/HIREX.SA2/FILENAME")
            pydoocs.write(
                "XFEL.UTIL/DYNPROP/HIREX.SA2/TIMESTAMP", self.date_time_obj)
            self.timestamp_doocs = pydoocs.read(
                "XFEL.UTIL/DYNPROP/HIREX.SA2/TIMESTAMP")
            self.ui.output.setText(self.ui.output.text(
                            ) + "DOOCS PIXEL_CALIBRATION value: " + str(self.pixel_doocs['data']) + '\n')
            self.ui.output.setText(self.ui.output.text(
                            ) + "DOOCS CENTRAL_ENERGY value: " + str(self.central_doocs['data']) + '\n')
            self.ui.output.setText(self.ui.output.text(
                            ) + "DOOCS FILENAME value: " + str(self.filename_doocs['data']) + '\n')
            self.ui.output.setText(self.ui.output.text(
                            ) + "DOOCS TIMESTAMP value: " + str(self.timestamp_doocs['data']) + '\n')
        except:
            self.doocs_permit = False
        if not self.doocs_permit:
            self.ui.output.setText(self.ui.output.text(
                            ) + "Control: no permission to write to DOOCS" + '\n')

    def load_from_doocs(self):
        try:
            self.pixel_doocs = pydoocs.read(
                            "XFEL.UTIL/DYNPROP/HIREX.SA2/PIXEL_CALIBRATION")
            self.central_doocs = pydoocs.read(
                            "XFEL.UTIL/DYNPROP/HIREX.SA2/CENTRAL_ENERGY")
            if self.central_doocs['data'] > 1999 and self.central_doocs['data'] <= 20000:
                self.parent.ui.sb_E0.setValue(self.central_doocs['data'])
            else:
                self.ui.output.setText(self.ui.output.text(
                                    ) + "Cannot set the Eo parameter outside the predefined range [2k eV, 20 keV]" + '\n')
            if self.pixel_doocs['data'] > 0 and self.pixel_doocs['data'] <= 1:
                self.parent.ui.sb_ev_px.setValue(self.pixel_doocs['data'])
            else:
                self.ui.output.setText(self.ui.output.text(
                                    ) + "Cannot set the ev/px parameter outside the range [0, 1]" + '\n')
        except:
            self.ui.output.setText(self.ui.output.text(
                                ) + "No permission to read from DOOCS" + '\n')

    def open_file(self):  # self.parent.data_dir
        self.pathname, _ = QtGui.QFileDialog.getOpenFileName(
            self, "Open Correlation Data", self.data_dir, 'txt (*.npz)', None, QtGui.QFileDialog.DontUseNativeDialog)
        if self.pathname != "":
            filename = os.path.basename(self.pathname)
            self.ui.file_name.setText(filename)
            self.load_corr2d()
        else:
            self.ui.file_name.setText('')
            #self.ui.output.setText('')

    def get_latest_npz(self):
        # * means all if need specific format then *.csv
        list_of_files = glob.glob(
            self.data_dir + "*_cor2d.npz")
        self.pathname = max(list_of_files, key=os.path.getmtime)
        #self.pathname = max(list_of_files, key=os.path.getctime)
        self.ui.file_name.setText(os.path.basename(self.pathname))
        print(self.pathname)
        self.load_corr2d()

    def load_corr2d(self):
        self.tt = np.load(self.pathname)
        self.orig_image = self.tt['corr2d']
        self.doocs_scale = self.tt['doocs_scale']
        if len(self.doocs_scale) != len(self.orig_image):
            self.np_doocs = self.doocs_scale[:-1]
        else:
            self.np_doocs = self.doocs_scale
        self.np_phen = self.tt['phen_scale']
        self.doocs_label = self.tt['doocs_channel']
        self.info_mono_no()

    def info_mono_no(self):
        if "XFEL.FEL/UNDULATOR.SASE2/MONOPA.2252.SA2/ANGLE" in self.doocs_label or "XFEL.FEL/UNDULATOR.SASE2/MONOPA.2307.SA2/ANGLE" in self.doocs_label:
            if "XFEL.FEL/UNDULATOR.SASE2/MONOPA.2252.SA2/ANGLE" in self.doocs_label:
                self.mono_no = 1
                try:
                    filedata = np.loadtxt(
                        self.pathname+'_status.txt', dtype='str', delimiter=',', skiprows=1)
                    pa_pos = np.where(
                        filedata == 'XFEL.FEL/UNDULATOR.SASE2/MONOPA.2252.SA2/ANGLE')
                    ra_pos = np.where(
                        filedata == 'XFEL.FEL/UNDULATOR.SASE2/MONORA.2252.SA2/ANGLE')
                    pa_row = pa_pos[0][0]
                    self.set_pitch_angle = float(filedata[pa_row][1])
                    ra_row = ra_pos[0][0]
                    self.set_roll_angle = float(filedata[ra_row][1])
                    self.ui.roll_angle.setValue(self.set_roll_angle)
                    self.ui.status.setText(
                        'Monochromator 1 image found; \nMachine status file found: roll angle=' + str(np.round(self.set_roll_angle, 4)) + ' deg \n')
                except:
                    self.set_roll_angle = self.ui.roll_angle.value()
                    self.set_pitch_angle = (
                        max(self.np_doocs)-min(self.np_doocs)/2)
                    self.ui.status.setText(
                        'Monochromator 1 image found; Machine status file not found.\n')
            elif "XFEL.FEL/UNDULATOR.SASE2/MONOPA.2307.SA2/ANGLE" in self.doocs_label:
                self.mono_no = 2
                try:
                    filedata = np.loadtxt(
                        self.pathname+'_status.txt', dtype='str', delimiter=',', skiprows=1)
                    pa_pos = np.where(
                        filedata == 'XFEL.FEL/UNDULATOR.SASE2/MONOPA.2307.SA2/ANGLE')
                    ra_pos = np.where(
                        filedata == 'XFEL.FEL/UNDULATOR.SASE2/MONORA.2307.SA2/ANGLE')
                    pa_row = pa_pos[0][0]
                    self.set_pitch_angle = float(filedata[pa_row][1])
                    ra_row = ra_pos[0][0]
                    self.set_roll_angle = float(filedata[ra_row][1])
                    self.ui.roll_angle.setValue(self.set_roll_angle)
                    self.ui.status.setText(
                        'Monochromator 2 image found; \nMachine status file found: roll angle=' + str(np.round(self.set_roll_angle, 4)) + ' deg \n')
                except:
                    self.set_roll_angle = self.ui.roll_angle.value()
                    self.set_pitch_angle = (
                        max(self.np_doocs)-min(self.np_doocs)/2)
                    self.ui.status.setText(
                        'Monochromator 2 image found; Machine status file not found.\n')
        else:
            self.ui.status.setText('Invalid input\n')


def main():

    #make pyqt threadsafe
    QtCore.QCoreApplication.setAttribute(QtCore.Qt.AA_X11InitThreads)
    #create the application
    app = QApplication(sys.argv)

    window = UICalculator()

    path = os.path.join(os.path.dirname(
        sys.modules[__name__].__file__), 'gui/hirex.png')
    app.setWindowIcon(QtGui.QIcon(path))
    window.show()
    window.raise_()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
