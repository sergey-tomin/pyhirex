"""
Sergey Tomin, DESY, 2021
"""
import sys
import numpy as np
import json
from PyQt5 import QtGui, QtCore
from PyQt5.QtWidgets import QWidget, QMessageBox, QApplication
from PyQt5 import QtWidgets
from gui.UILogger import Ui_Form
from gui.spectr_gui import send_to_desy_elog

import pyqtgraph as pg
from gui.UILogger import Ui_Form
import time
import os
from threading import Thread, Event
import logging

import math
import subprocess

# filename="logs/afb.log",
#logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

global first_maxspec_vals
global first_maxspec_av
global first_spec_fwhm_ev
global first_spec_peak_ev
global first_XGM_vals
global first_spec_integ_vals


def firstNonNan(listfloats):
    for item in listfloats:
        if math.isnan(item) == False:
            return item

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

class UILogger(QWidget):
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
        self.add_plot()
        self.timer_live = pg.QtCore.QTimer()
        self.timer_live.timeout.connect(self.plot_data)
        # self.ui.combo_log_ch_a.addItems(["max(spectrum)"])
        # self.ui.combo_log_ch_b.addItems(["max(av_spectrum)"])
        self.ui.pb_start_log.clicked.connect(self.start_stop_logger)
        #self.ui = self.parent.ui

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
        self.stop_logger()

    def start_stop_logger(self):
        if self.ui.pb_start_log.text() == "Stop":
            self.stop_logger()
        else:
            if self.parent.ui.pb_start.text() == "Start":
                self.error_box("Start spectrometer first")
                return
            if not self.parent.spectrometer.is_online():
                self.error_box("Spectrometer is not ONLINE")
                return
            #self.counter_spect = 0
            #self.data_2d = np.zeros((self.spectrometer.num_px, self.sb_2d_hist_size))
            self.maxspec_vals = []
            self.maxspec_av_vals = []
            self.spec_fwhm_ev_vals = []
            self.spec_peak_ev_vals = []
            self.XGM_vals = []
            self.spec_integ_vals = []

            self.timer_live.start(100)
            self.ui.pb_start_log.setText("Stop")

            self.ui.pb_start_log.setStyleSheet("color: rgb(63, 191, 95); font-size: 18pt")


    def plot_data(self):
        av_spec = self.parent.ave_spectrum
        if len(self.maxspec_vals) >= self.ui.sb_nponts.value():
            self.maxspec_vals = self.maxspec_vals[1:]
            self.maxspec_av_vals = self.maxspec_av_vals[1:]
            self.spec_fwhm_ev_vals = self.spec_fwhm_ev_vals[1:]
            self.spec_peak_ev_vals = self.spec_peak_ev_vals[1:]
            self.XGM_vals = self.XGM_vals[1:]
            self.spec_integ_vals = self.spec_integ_vals[1:]

        #x = np.arange(len(maxspec_av))
        # if self.ui.combo_log_ch_a.currentIndex() == 0:
            # self.single.setData(x=x, y=self.line1)
        # if self.ui.combo_log_ch_b.currentIndex() == 0:

        #######
        global first_maxspec_vals
        global first_maxspec_av
        global first_spec_fwhm_ev
        global first_spec_peak_ev
        global first_XGM_vals
        global first_spec_integ_vals

        #spec peak single checkbox
        if self.ui.chkbx_spec_pk_single.isChecked():
            try:
                maxspec_val = np.amax(self.parent.spectrum_event_disp)
            except ValueError:
                maxspec_val = 0

            self.maxspec_vals = np.append(self.maxspec_vals, maxspec_val)

            ### if chkbx_rel_changes is checked (the same blocks of code are in the next checkboxes)
            #check the first value non a Nan value in the array is chkbx_rel_changes was or at the start of the Logger
            if self.ui.chkbx_rel_changes.isDown() or np.size(self.maxspec_vals)==1:
                first_maxspec_vals = firstNonNan(self.maxspec_vals)

            if self.ui.chkbx_rel_changes.isChecked():
                #then try to apply first_maxspec_av to normalize on relative changes
                try:
                    self.maxspec_plot.setData(x=np.flip(np.arange(np.size(self.maxspec_vals))), y=self.maxspec_vals/first_maxspec_vals)

                #rise exception and agafind the first non a Nan value in the array
                except NameError:
                    first_maxspec_vals = firstNonNan(self.maxspec_vals)
                    self.maxspec_plot.setData(x=np.flip(np.arange(np.size(self.maxspec_vals))), y=self.maxspec_vals/first_maxspec_vals)
            else:
                self.maxspec_plot.setData(x=np.flip(np.arange(np.size(self.maxspec_vals))), y=self.maxspec_vals)
            ###
        else:
            self.maxspec_vals = np.append(self.maxspec_vals, np.nan)
            self.maxspec_plot.setData(x=np.flip(np.arange(np.size(self.maxspec_vals))), y=self.maxspec_vals)


        #######
        #spec peak average checkbox
        if self.ui.chkbx_spec_pk_average.isChecked():
            try:
                maxspec_av = np.amax(av_spec)
            except ValueError:
                maxspec_av = 0

            self.maxspec_av_vals = np.append(self.maxspec_av_vals, maxspec_av)

            if self.ui.chkbx_rel_changes.isDown() or np.size(self.maxspec_av_vals)==1:
                first_maxspec_av = firstNonNan(self.maxspec_av_vals)

            if self.ui.chkbx_rel_changes.isChecked():
                try:
                    self.average.setData(x=np.flip(np.arange(np.size(self.maxspec_av_vals))), y=self.maxspec_av_vals/first_maxspec_av)

                except NameError:
                    first_maxspec_av = firstNonNan(self.maxspec_av_vals)
                    self.average.setData(x=np.flip(np.arange(np.size(self.maxspec_av_vals))), y=self.maxspec_av_vals/first_maxspec_av)
            else:
                self.average.setData(x=np.flip(np.arange(np.size(self.maxspec_av_vals))), y=self.maxspec_av_vals)

        else:
            self.maxspec_av_vals = np.append(self.maxspec_av_vals, np.nan)
            self.average.setData(x=np.flip(np.arange(np.size(self.maxspec_av_vals))), y=self.maxspec_av_vals)


        #######
        #BW checkbox
        if self.ui.chkbx_spec_BW.isChecked():
            spec_fwhm_ev = self.parent.fwhm_ev
            self.spec_fwhm_ev_vals = np.append(self.spec_fwhm_ev_vals, spec_fwhm_ev)

            if self.ui.chkbx_rel_changes.isDown() or np.size(self.spec_fwhm_ev_vals)==1:
                first_spec_fwhm_ev = firstNonNan(self.spec_fwhm_ev_vals)

            if self.ui.chkbx_rel_changes.isChecked():
                try:
                    self.fwhm_ev_pos.setData(x=np.flip(np.arange(np.size(self.spec_fwhm_ev_vals))), y=self.spec_fwhm_ev_vals/first_spec_fwhm_ev)
                except NameError:
                    first_spec_fwhm_ev = firstNonNan(self.spec_fwhm_ev_vals)
                    self.fwhm_ev_pos.setData(x=np.flip(np.arange(np.size(self.spec_fwhm_ev_vals))), y=self.spec_fwhm_ev_vals/first_spec_fwhm_ev)
            else:
                self.fwhm_ev_pos.setData(x=np.flip(np.arange(np.size(self.spec_fwhm_ev_vals))), y=self.spec_fwhm_ev_vals)

        else:
            self.spec_fwhm_ev_vals = np.append(self.spec_fwhm_ev_vals, np.nan)
            self.fwhm_ev_pos.setData(x=np.flip(np.arange(np.size(self.spec_fwhm_ev_vals))), y=self.spec_fwhm_ev_vals)


        #######
        #spec pos checkbox
        if self.ui.chkbx_spec_pos.isChecked():
            spec_peak_ev = self.parent.peak_ev
            self.spec_peak_ev_vals = np.append(self.spec_peak_ev_vals, spec_peak_ev)

            if self.ui.chkbx_rel_changes.isDown() or np.size(self.spec_peak_ev_vals)==1:
                first_spec_peak_ev = firstNonNan(self.spec_peak_ev_vals)

            if self.ui.chkbx_rel_changes.isChecked():
                try:
                    self.peak_ev_pos.setData(x=np.flip(np.arange(np.size(self.spec_peak_ev_vals))), y=self.spec_peak_ev_vals/first_spec_peak_ev)
                except NameError:
                    first_spec_peak_ev = firstNonNan(self.spec_peak_ev_vals)
                    self.peak_ev_pos.setData(x=np.flip(np.arange(np.size(self.spec_peak_ev_vals))), y=self.spec_peak_ev_vals/first_spec_peak_ev)

            else:
                self.peak_ev_pos.setData(x=np.flip(np.arange(np.size(self.spec_peak_ev_vals))), y=self.spec_peak_ev_vals)

        else:
            self.spec_peak_ev_vals = np.append(self.spec_peak_ev_vals, np.nan)
            self.peak_ev_pos.setData(x=np.flip(np.arange(np.size(self.spec_peak_ev_vals))), y=self.spec_peak_ev_vals)


        #######
        #XGM signal checkbox
        if self.ui.chkbx_XGM_integral.isChecked():
            self.XGM_vals = np.append(self.XGM_vals, self.parent.xgm.get_value())

            if self.ui.chkbx_rel_changes.isDown() or np.size(self.XGM_vals)==1:
                first_XGM_vals = firstNonNan(self.XGM_vals)

            if self.ui.chkbx_rel_changes.isChecked():
                try:
                    self.XGM.setData(x=np.flip(np.arange(np.size(self.XGM_vals))), y=self.XGM_vals/first_XGM_vals)
                except NameError:
                    first_XGM_vals = firstNonNan(self.XGM_vals)
                    self.XGM.setData(x=np.flip(np.arange(np.size(self.XGM_vals))), y=self.XGM_vals/first_XGM_vals)

            else:
                self.XGM.setData(x=np.flip(np.arange(np.size(self.XGM_vals))), y=self.XGM_vals)

        else:
            self.XGM_vals = np.append(self.XGM_vals, np.nan)
            self.XGM.setData(x=np.flip(np.arange(np.size(self.XGM_vals))), y=self.XGM_vals)

        #######
        #spectrometer signal integral checkbox
        if self.ui.chkbx_spec_integral.isChecked():
            self.spec_integ_vals = np.append(self.spec_integ_vals, self.parent.ave_integ)

            if self.ui.chkbx_rel_changes.isDown() or np.size(self.spec_integ_vals)==1:
                first_spec_integ_vals = firstNonNan(self.spec_integ_vals)
                print('aaaa')

            if self.ui.chkbx_rel_changes.isChecked():
                try:
                    self.spec_integral.setData(x=np.flip(np.arange(np.size(self.spec_integ_vals))), y=self.spec_integ_vals/first_spec_integ_vals)
                except NameError:
                    first_spec_integ_vals = firstNonNan(self.spec_integ_vals)
                    self.spec_integral.setData(x=np.flip(np.arange(np.size(self.spec_integ_vals))), y=self.spec_integ_vals/first_spec_integ_vals)

            else:
                self.spec_integral.setData(x=np.flip(np.arange(np.size(self.spec_integ_vals))), y=self.spec_integ_vals)

        else:
            self.spec_integ_vals = np.append(self.spec_integ_vals, np.nan)
            self.spec_integral.setData(x=np.flip(np.arange(np.size(self.spec_integ_vals))), y=self.spec_integ_vals)

    def add_plot(self):

        gui_index = self.parent.ui.get_style_name_index()

        if "standard" in self.parent.gui_styles[gui_index]:
            pg.setConfigOption('background', 'w')
            pg.setConfigOption('foreground', 'k')
            single_pen = pg.mkPen("k")
        else:
            single_pen = pg.mkPen("w")

        # win = pg.GraphicsLayoutWidget() # justify='right',,
        win = pg.GraphicsView()
        win.setWindowTitle('logger')
        win.show()
        
        # self.label = pg.LabelItem(justify='left', row=0, col=0)
        # win.addItem(self.label)

        l = pg.GraphicsLayout()
        win.setCentralWidget(l)

        # add axis to layout
        ## at least one plotitem is used whioch holds its own viewbox and left axis
        pI = pg.PlotItem()
        v1 = pI.vb # reference to viewbox of the plotitem
        l.addItem(pI, row = 2, col = 3,  rowspan=1, colspan=1) # add plotitem to layout
        pI.getAxis('left').enableAutoSIPrefix(enable=True)  # stop the auto unit scaling on y axes
        pI.hideAxis('left')
        pI.showAxis('right')
        pI.getAxis("right").setLabel('Signal, arb.un', color='#ff0000')
        # pI.setLabel("left", 'Signal, arb.un', color='#ff0000')
        pI.setLabel('bottom', 'N of shots', color='#000000')

        # pI.showGrid(1, 1, 1)
        # self.vLine = pg.InfiniteLine(angle=90, movable=False)
        # self.hLine = pg.InfiniteLine(angle=0, movable=False)
        # pg.PlotItem().addItem(self.vLine, ignoreBounds=False)
        # pg.PlotItem().addItem(self.hLine, ignoreBounds=True)
        # a2 = pg.AxisItem("left")

        # a3.enableAutoSIPrefix(enable=True)  # stop the auto unit scaling on y axes
        # a4.enableAutoSIPrefix(enable=True)  # stop the auto unit scaling on y axes

        #invert axis
        v3 = pg.ViewBox(invertX=True)
        v4 = pg.ViewBox(invertX=True)
        v5 = pg.ViewBox(invertX=True)

        a3 = pg.AxisItem("right")
        pI.layout.addItem(a3, 2, 3)
        pI.scene().addItem(v3)

        a4 = pg.AxisItem("right")
        pI.layout.addItem(a4, 2, 4)
        pI.scene().addItem(v4)

        a5 = pg.AxisItem("left")
        pI.layout.addItem(a5, 2, 0)
        pI.scene().addItem(v5)

        v1.invertX(True)

        v1.setLimits(xMin=0, yMin=0)
        v3.setLimits(xMin=0)
        v4.setLimits(xMin=0)
        v5.setLimits(xMin=0)

         ###### very important lines
        layout = QtGui.QGridLayout()
        self.ui.widget_log.setLayout(layout)
        layout.addWidget(win, 0, 0)
        ######

        # add viewboxes to layout
        l.scene().addItem(v3)
        l.scene().addItem(v4)
        l.scene().addItem(v5)

        # link axis with viewboxes
        a3.linkToView(v3)
        a4.linkToView(v4)
        a5.linkToView(v5)

        # link viewboxes
        v3.setXLink(pI)
        v4.setXLink(pI)
        v5.setXLink(pI)

        a3.setLabel('FWHM, eV', color='#008000')
        a4.setLabel('Peak position, eV', color='#0000ff')
        a5.setLabel('XGM, mJ', color='#800080')

        # slot: update view when resized
        def updateViews():
            v3.setGeometry(v1.sceneBoundingRect())
            v4.setGeometry(v1.sceneBoundingRect())
            v5.setGeometry(v1.sceneBoundingRect())
        # plot
        self.maxspec_plot = pg.PlotCurveItem(pen=single_pen, name='maxspec_single_plot')

        pen1 = pg.mkPen('#ff0000', width=3)
        self.average =      pg.PlotCurveItem(pen=pen1, name='average')

        pen2 = pg.mkPen('#008000', width=2)
        self.fwhm_ev_pos =  pg.PlotCurveItem(pen=pen2, name='fwhm')

        pen3 = pg.mkPen('#0000ff', width=2)
        self.peak_ev_pos =  pg.PlotCurveItem(pen=pen3, name='peak_pos')
        
        pen4 = pg.mkPen('#800080', width=2)
        self.XGM =  pg.PlotCurveItem(pen=pen4, name='XGM')
        
        pen5 = pg.mkPen('#ffa500', width=2)
        self.spec_integral =  pg.PlotCurveItem(pen=pen5, name='spec_integral')

        v1.addItem(self.maxspec_plot)
        v1.addItem(self.average)
        v3.addItem(self.fwhm_ev_pos)
        v4.addItem(self.peak_ev_pos)
        v5.addItem(self.XGM)
        v5.addItem(self.spec_integral)

        # updates when resized
        v1.sigResized.connect(updateViews)
        # autorange once to fit views at start
        # v2.enableAutoRange(axis= pg.ViewBox.XYAxes, enable=True)
        # if self.ui.chkbx_rel_changes.isCheckable():
        v3.enableAutoRange(axis=pg.ViewBox.XYAxes, enable=True)
        v4.enableAutoRange(axis=pg.ViewBox.XYAxes, enable=True)
        v5.enableAutoRange(axis=pg.ViewBox.XYAxes, enable=True)

        updateViews()

    def stop_logger(self):
        self.timer_live.stop()
        logger.info("Stop Logger")
        self.timer_live.stop()
        self.ui.pb_start_log.setStyleSheet("color: rgb(255, 0, 0); font-size: 18pt")
        self.ui.pb_start_log.setText("Start")




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
        if reply==QtGui.QMessageBox.Yes:
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
        res = send_to_desy_elog(author="", title="pySpectrometer logger "+ device, severity="INFO", text=text, elog=self.Form.mi.logbook_name,
                          image=screenshot)
        if not res:
            self.Form.error_box("error during eLogBook sending")


def main():

    #make pyqt threadsafe
    QtCore.QCoreApplication.setAttribute(QtCore.Qt.AA_X11InitThreads)
    #create the application
    app = QApplication(sys.argv)

    window = UILogger()

    #show app
    #window.setWindowIcon(QtGui.QIcon('gui/angry_manul.png'))
    # setting the path variable for icon
    path = os.path.join(os.path.dirname(sys.modules[__name__].__file__), 'gui/manul.png')
    app.setWindowIcon(QtGui.QIcon(path))
    window.show()
    window.raise_()
    #Build documentaiton if source files have changed
    #os.system("cd ./docs && xterm -T 'Ocelot Doc Builder' -e 'bash checkDocBuild.sh' &")
    #exit script
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
