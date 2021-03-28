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
        self.ui.combo_log_ch_a.addItems(["max(spectrum)"])
        self.ui.combo_log_ch_b.addItems(["max(av_spectrum)"])
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
        pass

    def start_stop_logger(self):
        if self.ui.pb_start_log.text() == "Stop":
            self.timer_live.stop()
            self.ui.pb_start_log.setStyleSheet("color: rgb(255, 0, 0); font-size: 18pt")
            self.ui.pb_start_log.setText("Start")
        else:
            if self.parent.ui.pb_start.text() == "Start":
                self.error_box("Start spectrometer first")
                return
            if not self.parent.spectrometer.is_online():
                self.error_box("Spectrometer is not ONLINE")
                return
            #self.counter_spect = 0
            #self.data_2d = np.zeros((self.spectrometer.num_px, self.sb_2d_hist_size))
            self.line1 = []
            self.line2 = []
            self.line3 = []
            self.line4 = []

            self.timer_live.start(100)
            self.ui.pb_start_log.setText("Stop")

            self.ui.pb_start_log.setStyleSheet("color: rgb(63, 191, 95); font-size: 18pt")

    def plot_data(self):
        spec = self.parent.spectrum_event_disp
        av_spec = self.parent.ave_spectrum

        if len(self.line1) >= self.ui.sb_nponts.value():
            self.line1 = self.line1[1:]
            self.line2 = self.line2[1:]
            self.line3 = self.line3[1:]
            self.line4 = self.line4[1:]

        self.line1 = np.append(self.line1, np.max(spec))
        self.line2 = np.append(self.line2, np.max(av_spec))
        x = np.arange(len(self.line1))
        if self.ui.combo_log_ch_a.currentIndex() == 0:
            self.single.setData(x=x, y=self.line1)
        if self.ui.combo_log_ch_b.currentIndex() == 0:
            self.average.setData(x=x, y=self.line2)


    def add_plot(self):
        gui_index = self.parent.ui.get_style_name_index()
        if "standard" in self.parent.gui_styles[gui_index]:
            pg.setConfigOption('background', 'w')
            pg.setConfigOption('foreground', 'k')
            single_pen = pg.mkPen("k")
        else:
            single_pen = pg.mkPen("w")

        win = pg.GraphicsLayoutWidget()
        # justify='right',,
        self.label = pg.LabelItem(justify='left', row=0, col=0)
        win.addItem(self.label)

        # self.plot1 = win.addPlot(row=0, col=0)
        self.plot1 = win.addPlot(row=1, col=0)

        self.label2 = pg.LabelItem(justify='right')
        win.addItem(self.label2, row=0, col=0)

        self.plot1.setLabel('left', "A", units='au')
        self.plot1.setLabel('bottom', "", units='eV')

        self.plot1.showGrid(1, 1, 1)

        self.plot1.getAxis('left').enableAutoSIPrefix(enable=False)  # stop the auto unit scaling on y axes
        layout = QtGui.QGridLayout()
        self.ui.widget_log.setLayout(layout)
        layout.addWidget(win, 0, 0)

        self.plot1.setAutoVisible(y=True)

        self.plot1.addLegend()

        self.single = pg.PlotCurveItem(pen=single_pen, name='single')

        self.plot1.addItem(self.single)

        pen = pg.mkPen((51, 255, 51), width=2)
        pen = pg.mkPen((255, 0, 0), width=3)
        # self.average = pg.PlotCurveItem(x=[], y=[], pen=pen, name='average')
        self.average = pg.PlotCurveItem(pen=pen, name='average')

        self.plot1.addItem(self.average)

        pen = pg.mkPen((0, 255, 255), width=2)

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

    def stop_statistics(self):
        self.stop_feedback()
        self.statistics_timer.stop()
        logger.info("Stop Statistics")
        self.pb_start_statistics.setStyleSheet("color: rgb(85, 255, 127);")
        self.pb_start_statistics.setText("Statistics Accum On")




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
