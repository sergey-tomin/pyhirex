"""
Christian Grech, DESY, 2021
based on logger.py by Sergey Tomin
"""
import sys
import numpy as np
import pathlib
from PyQt5 import QtGui, QtCore
from PyQt5.QtGui import QBrush, QColor
from PyQt5.QtWidgets import QWidget, QApplication
import pyqtgraph as pg
from gui.UICalculator import Ui_Form
import os
import glob
import logging
from matplotlib import cm
import pandas as pd
from scipy import ndimage
from scipy.spatial import distance
from scipy.optimize import fsolve
from skimage.filters import threshold_yen
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import NearestNeighbors, KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier, AdaBoostClassifier, RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import preprocessing
from scipy import interpolate
import re
from skimage.transform import hough_line, hough_line_peaks
from model_functions.HXRSS_Bragg_max_generator import HXRSS_Bragg_max_generator
from model_functions.HXRSS_Bragg_single import HXRSSsingle

#from model_functions.HXRSS_Bragg_generator import HXRSS_Bragg_generator
from itertools import cycle
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
        self.n, self.d_kernel, self.e_kernel = 0, 2, 2
        self.mode = 0
        self.mono_no = None
        #self.max_E = 2000
        #self.max_P = 2
        #self.slope_allowance = 25
        #self.intercept_allowance = 2000
        #self.max_distance = 1000
        self.hmax, self.kmax, self.lmax = 5, 5, 5
        self.img_corr2d = None
        self.min_phen, self.max_phen = 0, 0
        self.min_pangle, self.max_pangle = 0, 0
        self.dE_mean = 0
        self.nomatch = 0
        self.yvalue = []
        self.pitch_angle_range, self.min_angle_list, self.spec_data_list, self.slope_list, self.y_intercept_list, self.centroid_pa_list, self.centroid_phen_list, self.max_angle_list = [], [], [], [], [], [], [], []
        self.tngnt_slope_list, self.tngnt_intercept_list, self.tngnt_gid_list, self.tngnt_centroid_list, self.tngnt_centroid_y_list, self.tngnt_roll_angle_list, self.interp_Bragg_list = [], [], [], [], [], [], []
        self.detected_slope_list, self.detected_intercept_list, self.detected_id_list, self.detected_line_min_angle_list, self.detected_line_max_angle_list,  self.detected_line_roll_angle_list, self.actual_E, self.dE_list, self.ans_list, self.detected_centroid_x_list, self.detected_centroid_y_list = [], [], [], [], [], [], [], [], [], [], []
        self.h_list, self.k_list, self.l_list, self.roll_list, self.centroid_list = [], [], [], [], []
        self.ind = ''
        DIR_NAME = os.path.basename(pathlib.Path(__file__).parent.absolute())
        self.path = path[:path.find(DIR_NAME)]
        self.data_dir = path[:path.find(
            "user")] + "user" + os.sep + PY_SPECTROMETER_DIR + os.sep + "SASE2" + os.sep
        print(self.data_dir)

        self.ui.pb_start_calc.clicked.connect(self.start_stop_calc_from_npz)
        self.ui.browse_button.clicked.connect(self.open_file)
        self.ui.file_name.setText('')
        self.ui.roll_angle.setDecimals(4)
        self.ui.roll_angle.setSuffix(" °")
        self.ui.roll_angle.setRange(0, 2)
        self.ui.roll_angle.setValue(1.5013)
        self.ui.roll_angle.setSingleStep(0.001)
        self.ui.tableWidget.setRowCount(0)

        # Set up and show the two graph axes
        self.add_image_widget()
        self.add_plot_widget()
        self.get_latest_npz()

        #self.ui = self.parent.ui

    def reset(self):

        #self.text.setText('')
        self.dE_mean, self.min_phen, self.max_phen = 0, 0, 0
        self.min_pangle, self.max_pangle = 0, 0
        self.dE_mean = 0
        self.ind = ''
        self.pitch_angle_range, self.min_angle_list, self.spec_data_list, self.slope_list, self.y_intercept_list, self.centroid_pa_list, self.centroid_phen_list, self.max_angle_list = [], [], [], [], [], [], [], []
        self.tngnt_slope_list, self.tngnt_intercept_list, self.tngnt_gid_list, self.tngnt_centroid_list, self.tngnt_centroid_y_list, self.tngnt_roll_angle_list, self.interp_Bragg_list = [], [], [], [], [], [], []
        self.detected_slope_list, self.detected_intercept_list, self.detected_id_list, self.detected_line_min_angle_list, self.detected_line_max_angle_list,  self.detected_line_roll_angle_list, self.dE_list, self.ans_list, self.detected_centroid_x_list, self.detected_centroid_y_list, self.actual_E = [], [], [], [], [], [], [], [], [], [], []
        self.h_list, self.k_list, self.l_list, self.roll_list, self.pa, self.phen, self.gid_list = [
            ], [], [], [], [], [], []
        self.ui.tableWidget.setRowCount(0)
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
        self.img_corr2d.setLabel('left', "E_ph", units='eV')
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
        self.plot1.setYLink(self.img_corr2d)

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
            if self.phen[r][100] <= max(self.np_phen)+700 and self.phen[r][100] >= min(self.np_phen)-700:
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
        self.ind = ''

    def binarization(self):
        # all values below 0 threshold are set to 0
        self.phen_res = self.np_phen[2] - self.np_phen[1]

        self.min_pangle = min(self.np_doocs)
        self.max_pangle = max(self.np_doocs)
        self.corr2d[self.corr2d < 0] = 0
        #self.image = self.corr2d.T
        #thresh = threshold_yen(self.image, nbins=256)
        #binary = self.image > thresh
        #self.processed_image = binary
        #### ALTERNATE MANUAL THRESHOLDING
        range_scale = np.ptp(self.corr2d)
        threshold = 0.12 * range_scale
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
            self.DTHR = 0.1675
            self.alpha = 0.00238
        else:
            self.DTHP = -0.392
            self.dthy = 1.17
            self.DTHR = 0.1675
            self.alpha = 0.00238
        self.pa_range = np.linspace(self.min_pangle-1, self.max_pangle+1, 200)
        self.pa_range_plot = np.linspace(
            self.min_pangle-1, self.max_pangle+1, 200)
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
                if y1[0] < max(self.np_phen)+150 and y1[0] > min(self.np_phen)-150:
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
        clf = RandomForestClassifier(n_estimators=20, random_state=1)
        clf.fit(X, y)
        self.df_test['gid'] = clf.predict(self.df_test_scaled)
        self.df_detected = pd.DataFrame(dict(slope=self.df_test['slope'], intercept=self.df_test['intercept'], min_angle=self.df_test['min_angle'], max_angle=self.df_test['max_angle'],
                                             gid=self.df_test['gid'], roll_angle=self.df_test['roll_angle'], centroid_pa=self.df_test['centroid_pa'], centroid_phen=self.df_test['centroid_phen']))
        #

    def dispersion_cal(self):
        for slope, mdl_slope, curve_id, centroid_pa in zip(self.df_detected['slope'], self.df_detected['mdl_slope'], self.df_detected['gid'], self.df_detected['centroid_pa']):
            msg = 'Id:' + curve_id + ' matched to line with centroid: ' + \
                str(np.round(centroid_pa, 1)) + ' deg\n'
            self.ui.output.setText(self.ui.output.text() + msg)
            if abs(mdl_slope/slope) > 1.25 or abs(mdl_slope/slope) < 0.75:
                self.ind = 'error'
            self.add_table_row(curve_id + 'ev/px', str(np.round(self.scale_yaxis, 3)), str(
                np.round(self.scale_yaxis*mdl_slope/slope, 3)))

    def hkl_roll_separator(self):
        for gid_item, roll, cent_x in zip(self.df_detected['gid'], self.df_detected['roll_angle'], self.df_detected['centroid_pa']):
            num = [int(s) for s in re.findall(r'-?\d+', str(gid_item))]
            self.h_list.append(num[0])
            self.k_list.append(num[1])
            self.l_list.append(num[2])
            self.roll_list.append(roll)
            self.centroid_list.append(cent_x-self.DTHP)

    def offset_calc_and_plot(self):
        self.roll_list = [self.set_roll_angle]
        self.phen, self.pa, gid_list, _roll_list, self.color_list, self.linestyle_list = HXRSS_Bragg_max_generator(
            self.pa_range_plot, self.hmax, self.kmax, self.lmax, self.DTHP, self.dthy, self.roll_list, self.DTHR, self.alpha)

        pa_dE, phen_Actual, gid_list_s, model_slope_list = HXRSSsingle(
            (self.h_list, self.k_list, self.l_list, self.roll_list, self.centroid_list), self.DTHP, self.dthy, self.DTHR, self.alpha)

        df_offset = pd.DataFrame(
            dict(E_model=phen_Actual, gid=gid_list_s, centroid_pa=pa_dE, mdl_slope=model_slope_list))
        self.df_detected = self.df_detected.merge(
            df_offset, on=['gid', 'centroid_pa'], how='left')
        self.df_detected['dE'] = self.df_detected['E_model'] - \
            self.df_detected['centroid_phen']
        self.dispersion_cal()
        self.dE_mean = np.mean(self.df_detected['dE'])
        if np.isnan(self.dE_mean) is True:
            self.dE_mean = 0
        print('E_offset is'+str(self.dE_mean))
        #self.E_actual_mean = np.mean(self.df_detected['actual_E'])
        self.add_plot()
        self.plot1.setYRange(min(self.np_phen)+self.dE_mean,
                             max(self.np_phen)+self.dE_mean, padding=None, update=True)
        for E, id in zip(self.df_detected['dE'], self.df_detected['gid']):
            #self.add_text_to_plot(x, max(self.np_phen)-10, E)
            if abs(E) > 200:
                self.ind = 'error'
            self.add_table_row(id + ' Eoff', '-',
                               str(np.round(E, 1))+' eV')
        #self.ui.output.setText(self.ui.output.text() + 'Average Energy Offset: '
            #+ str(np.round(self.dE_mean, 1))+' eV\n')
        if abs(self.dE_mean) > 200:
            self.ind = 'error'
        self.add_table_row(
            'Avg. Eoff', '-', str(np.round(self.dE_mean, 1))+' eV')
        #self.ui.output.setText(self.ui.output.text() + 'Currently set Central Energy: ' + str(np.round(self.parent.ui.sb_E0.value(), 0)) + 'eV, Proposed calibrated Central Energy: '
        #+ str(np.round((self.parent.ui.sb_E0.value()+self.dE_mean), 0))+' eV\n')
        self.add_table_row('Eo', str(np.round(self.parent.ui.sb_E0.value(
        ), 0)) + ' eV', str(np.round((self.parent.ui.sb_E0.value()+self.dE_mean), 0))+' eV')
        self.add_table_row(' ', ' ', ' ')

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

    def img_processing(self):
        self.processed_image = ndimage.grey_dilation(
            self.processed_image, size=(self.d_kernel, self.d_kernel))
        self.processed_image = ndimage.grey_erosion(
            self.processed_image, size=(self.e_kernel, self.e_kernel))

    def add_text_to_plot(self, x, y, E):
        self.text = pg.TextItem(color='w')
        self.img_corr2d.addItem(self.text)
        self.text.setText(str(np.round(E, 1)) + ' eV')
        self.text.setPos(x, y)
        self.text.setZValue(5)
        self.text.show()

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
            #self.ui.output.setText('')

    def get_latest_npz(self):
        # * means all if need specific format then *.csv
        #list_of_files = glob.glob(
        #    self.data_dir + "*_cor2d.npz")
        list_of_files = glob.glob(
            '/Users/christiangrech/Nextcloud/Notebooks/HXRSS/Data/npz/' + "*_cor2d.npz")
        self.pathname = max(list_of_files, key=os.path.getmtime)
        #self.pathname = max(list_of_files, key=os.path.getctime)
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
