"""
Sergey Tomin, DESY, 2021
"""
import sys
import numpy as np
import json
from PyQt5 import QtGui, QtCore
from PyQt5.QtWidgets import QWidget, QMessageBox, QApplication
import pyqtgraph as pg
from gui.UILogger import Ui_Form
import time
import os
from threading import Thread, Event
import logging

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
            self.line2 = []
            self.line3 = []
            self.line4 = []

            self.timer_live.start(100)
            self.ui.pb_start_log.setText("Stop")

            self.ui.pb_start_log.setStyleSheet("color: rgb(63, 191, 95); font-size: 18pt")

    def plot_data(self):
        from hirex import fwhm3   

        av_spec = self.parent.ave_spectrum

        if len(self.maxspec_vals) >= self.ui.sb_nponts.value():
            self.maxspec_vals = self.maxspec_vals[1:]
            self.line2 = self.line2[1:]
            self.line3 = self.line3[1:]
            self.line4 = self.line4[1:]
        
        
        try:
            maxspec_av = np.amax(av_spec)
        except ValueError:
            maxspec_av = 0
            
        self.line2 = np.append(self.line2, maxspec_av)
        
        self.parent.peak_ev = self.parent.x_axis_disp[np.argmax(self.parent.ave_spectrum)]
        try:
            p1interp, p2interp = fwhm3(np.array(self.parent.ave_spectrum))
            fwhm_px = p2interp - p1interp
            peak_px = (p2interp + p1interp)/2
        except ValueError:
            fwhm_px = 0
            peak_px = 0 
        px_ev = (self.parent.x_axis_disp[1] - self.parent.x_axis_disp[0])
        self.parent.peak_ev = self.parent.x_axis_disp[int(np.floor(peak_px))] + px_ev * (peak_px - np.floor(peak_px))
        self.parent.fwhm_ev = fwhm_px * px_ev

        #x = np.arange(len(maxspec_av))
        # if self.ui.combo_log_ch_a.currentIndex() == 0:
            # self.single.setData(x=x, y=self.line1)
        # if self.ui.combo_log_ch_b.currentIndex() == 0:
        
        self.average.setData(x=np.arange(np.size(self.line2)), y=self.line2)
        
        #spec peak single checkbox
        if self.ui.chkbx_spec_pk_single.isChecked():
            try:
                maxspec_val = np.amax(self.parent.spectrum_event_disp)
            except ValueError:
                maxspec_val = 0
            self.maxspec_vals = np.append(self.maxspec_vals, maxspec_val)
            self.maxspec_plot.setData(x=np.arange(np.size(self.maxspec_vals)), y=self.maxspec_vals)
        else:
            self.maxspec_vals = np.append(self.maxspec_vals, np.nan)
        
        # #spec peak average checkbox 
        # if self.ui.chkbx_spec_pk_average.isCheckable():
        #     try:
        #         pass
        #     except ValueError:
        #         pass
        # else:
        #     pass 

        #spec pos and BW checkboxes         
        if self.ui.chkbx_spec_pos.isChecked():
            self.line3 = np.append(self.line3, self.parent.fwhm_ev)
            self.fwhm_ev_pos.setData(x=np.arange(np.size(self.line3)), y=self.line3)
        else:
            self.line3 = np.append(self.line3, np.nan)
            
        if self.ui.chkbx_spec_BW.isChecked():
            self.line4 = np.append(self.line4, self.parent.peak_ev)
            self.peak_ev_pos.setData(x=np.arange(np.size(self.line4)), y=self.line4)
        else:
            self.line4 = np.append(self.line4, np.nan)
            
            
    def add_plot(self):
        
        gui_index = self.parent.ui.get_style_name_index()
        
        if "standard" in self.parent.gui_styles[gui_index]:
            pg.setConfigOption('background', 'w')
            pg.setConfigOption('foreground', 'k')
            single_pen = pg.mkPen("k")
        else:
            single_pen = pg.mkPen("w")

        win = pg.GraphicsLayoutWidget() # justify='right',,
        self.label = pg.LabelItem(justify='left', row=0, col=0)
        win.addItem(self.label)
        
        a2 = pg.AxisItem("left")
        a3 = pg.AxisItem("left")
        a4 = pg.AxisItem("left")
            
        v2 = pg.ViewBox()
        v3 = pg.ViewBox()
        v4 = pg.ViewBox()
        
        # add axis to layout
        ## watch the col parameter here for the position
        win.addItem(a2, row = 2, col = 5,  rowspan=1, colspan=1)
        win.addItem(a3, row = 2, col = 4,  rowspan=1, colspan=1)
        win.addItem(a4, row = 2, col = 3,  rowspan=1, colspan=1)
     
        # plotitem and viewbox
        ## at least one plotitem is used whioch holds its own viewbox and left axis
        pI = pg.PlotItem()
        v1 = pI.vb # reference to viewbox of the plotitem
        win.addItem(pI, row = 2, col = 6,  rowspan=1, colspan=1) # add plotitem to layout
        
        
        pI.getAxis('left').enableAutoSIPrefix(enable=False)  # stop the auto unit scaling on y axes
        layout = QtGui.QGridLayout()
        self.ui.widget_log.setLayout(layout)
        layout.addWidget(win, 0, 0)
        
        pI.addLegend()

        # add viewboxes to layout 
        win.scene().addItem(v2)
        win.scene().addItem(v3)
        win.scene().addItem(v4)
        
        # link axis with viewboxes
        a2.linkToView(v2)
        a3.linkToView(v3)
        a4.linkToView(v4)
        
        # link viewboxes
        v2.setXLink(v1)
        v3.setXLink(v2)
        v4.setXLink(v3)
        
        # axes labels
        pI.getAxis("left").setLabel('maxspec_single_plot', color='#FFFFFF')
        a2.setLabel('average', color='#ff0000')
        a3.setLabel('fwhm', color='#008000')
        a4.setLabel('peak_pos', color='#0000ff')
        
        # slot: update view when resized
        def updateViews():
            v2.setGeometry(v1.sceneBoundingRect())
            v3.setGeometry(v1.sceneBoundingRect())
            v4.setGeometry(v1.sceneBoundingRect())
                
        # plot
        self.maxspec_plot = pg.PlotCurveItem(pen=single_pen, name='maxspec_single_plot')
        
        pen1 = pg.mkPen('#ff0000', width=3)
        self.average =      pg.PlotCurveItem(pen=pen1, name='average')
        
        pen2 = pg.mkPen('#008000', width=2)
        self.fwhm_ev_pos =  pg.PlotCurveItem(pen=pen2, name='fwhm')
        
        pen3 = pg.mkPen('#0000ff', width=2)
        self.peak_ev_pos =  pg.PlotCurveItem(pen=pen3, name='peak_pos')
    
        v1.addItem(self.maxspec_plot)
        v2.addItem(self.average)
        v3.addItem(self.fwhm_ev_pos)
        v4.addItem(self.peak_ev_pos)      
        
        # updates when resized
        v1.sigResized.connect(updateViews)
        
        # autorange once to fit views at start
        v2.enableAutoRange(axis= pg.ViewBox.XYAxes, enable=True)
        v3.enableAutoRange(axis= pg.ViewBox.XYAxes, enable=True)
        v4.enableAutoRange(axis= pg.ViewBox.XYAxes, enable=True)
        
        # updateViews()
        
    def add_plot_1(self):
        gui_index = self.parent.ui.get_style_name_index()
        
        if "standard" in self.parent.gui_styles[gui_index]:
            pg.setConfigOption('background', 'w')
            pg.setConfigOption('foreground', 'k')
            single_pen = pg.mkPen("k")
        else:
            single_pen = pg.mkPen("w")

        win = pg.GraphicsLayoutWidget() # justify='right',,
        self.label = pg.LabelItem(justify='left', row=0, col=0)
        win.addItem(self.label)
        
        # #self.plot1 = win.addPlot(row=0, col=0)
        self.plot1 = win.addPlot(row=1, col=0)

        self.label2 = pg.LabelItem(justify='right')
        win.addItem(self.label2, row=0, col=0)

        self.plot1.setLabel('left', "A", units='au')
        self.plot1.setLabel('bottom', "", units='n shots')

        self.plot1.showGrid(1, 1, 1)

        self.plot1.getAxis('left').enableAutoSIPrefix(enable=False)  # stop the auto unit scaling on y axes
        layout = QtGui.QGridLayout()
        self.ui.widget_log.setLayout(layout)
        layout.addWidget(win, 0, 0)

        self.plot1.setAutoVisible(y=True)

        self.plot1.addLegend()

        self.maxspec_plot = pg.PlotCurveItem(pen=single_pen, name='maxspec_single_plot')

        self.plot1.addItem(self.maxspec_plot)

        pen = pg.mkPen((51, 255, 51), width=2)
        pen = pg.mkPen((255, 0, 0), width=3)
        # self.average = pg.PlotCurveItem(x=[], y=[], pen=pen, name='average')
        self.average = pg.PlotCurveItem(pen=pen, name='average')

        self.plot1.addItem(self.average)

        # pen = pg.mkPen((0, 255, 255), width=2)
        
        self.fwhm_ev_pos = pg.PlotCurveItem(pen=pen, name='fwhm')
        
        # self.plot1.addItem(self.fwhm_ev_pos)

        pen = pg.mkPen((1, 5, 155), width=2)

        self.peak_ev_pos = pg.PlotCurveItem(pen=pen, name='peak')

        # self.plot1.addItem(self.peak_ev_pos)

        pen = pg.mkPen((25, 25, 25), width=2)
        
        self.fit_func = pg.PlotCurveItem(pen=pen, name='Gauss Fit')

        # self.plot1.addItem(self.fit_func)
        # self.plot1.enableAutoRange(False)
        # self.textItem = pg.TextItem(text="", border='w', fill=(0, 0, 0))
        # self.textItem.setPos(10, 10)

        pen = pg.mkPen((0, 100, 0), width=1)
        # self.average = pg.PlotCurveItem(x=[], y=[], pen=pen, name='average')
        self.back_plot = pg.PlotCurveItem(pen=pen, name='background')

        # self.plot1.addItem(self.back_plot) ##################################### SS removed, as typically we don;t need it once start pySpectrometer

        # cross hair
        self.vLine = pg.InfiniteLine(angle=90, movable=False)
        self.hLine = pg.InfiniteLine(angle=0, movable=False)
        self.plot1.addItem(self.vLine, ignoreBounds=True)
        self.plot1.addItem(self.hLine, ignoreBounds=True)
        
        # self.plot1.sigRangeChanged.connect(self.zoom_signal)
        
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
