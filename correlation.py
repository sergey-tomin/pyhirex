import numpy as np
import pyqtgraph as pg
from threading import Thread, Event
from PyQt5 import QtGui, QtCore
import time
from mint.opt_objects import Device
from scipy import ndimage
from matplotlib import cm


class CorrelInterface:
    """
    Main class for orbit correction
    """
    def __init__(self, parent):
        self.parent = parent
        self.ui = self.parent.ui
        self.mi = self.parent.mi

        self.add_corel_plot()
        self.add_hist_plot()
        #self.plot1.scene().sigMouseMoved.connect(self.mouseMoved)
        self.ui.cb_corel_spect.addItem("Peak Pos")
        self.ui.cb_corel_spect.addItem("Peak Ampl")
        self.doocs_dev = None
        self.doocs_dev_hist = None
        self.get_device()
        self.ui.le_doocs_ch_cor.editingFinished.connect(self.get_device)
        self.ui.le_doocs_ch_hist.editingFinished.connect(self.get_device)
        self.plot_timer = pg.QtCore.QTimer()
        self.plot_timer.timeout.connect(self.plot_correl)
        self.plot_timer.start(100)

        self.plot_timer_hist = pg.QtCore.QTimer()
        self.plot_timer_hist.timeout.connect(self.plot_histogram)
        self.plot_timer_hist.start(100)

        self.peak = []
        self.doocs_vals = []
        self.doocs_vals_hist = []



        # self.ui.pb_start_scan.clicked.connect(self.start_stop_scan)
        # self.ui.pb_check_range.clicked.connect(self.check_range)
        # self.ui.pb_show_map.clicked.connect(self.show_hide_map)

    def get_device(self):
        if self.ui.is_le_addr_ok(self.ui.le_doocs_ch_cor):
            eid = self.ui.le_doocs_ch_cor.text()
            self.doocs_dev = Device(eid=eid)
            self.doocs_dev.mi = self.mi
        else:
            self.doocs_dev = None

        if self.ui.is_le_addr_ok(self.ui.le_doocs_ch_hist):
            eid = self.ui.le_doocs_ch_hist.text()
            self.doocs_dev_hist = Device(eid=eid)
            self.doocs_dev_hist.mi = self.mi
        else:
            self.doocs_dev_hist = None



    def plot_histogram(self):

        #current_mode = self.ui.cb_corel_spect.currentText()
        if self.ui.pb_start.text() == "Start" or self.parent.peak_ev is None:
            return
        if self.doocs_dev_hist is not None:
            #self.peak.insert(0, self.parent.peak_ev)
            self.doocs_vals_hist.insert(0, self.doocs_dev_hist.get_value())
            n_shots = int(self.ui.sb_n_shots_hist.value())
            if len(self.doocs_vals_hist) > n_shots:
                #self.peak = self.peak[:n_shots]
                self.doocs_vals_hist = self.doocs_vals_hist[:n_shots]

            if self.ui.scan_tab.currentIndex() == 1:
                val_min = np.min(self.doocs_vals_hist)
                val_max = np.max(self.doocs_vals_hist)
                y, x = np.histogram(self.doocs_vals_hist, bins=np.linspace(val_min, val_max, 50))
                self.plot_hist.plot(x, y, stepMode=True,  fillLevel=0,  brush=(0,0,255,150), clear=True)

    def plot_correl(self):

        current_mode = self.ui.cb_corel_spect.currentText()
        if self.ui.pb_start.text() == "Start" or self.parent.peak_ev is None:
            return
        if current_mode == "Peak Pos" and self.doocs_dev is not None:
            self.peak.insert(0, self.parent.peak_ev)
        
            self.doocs_vals.insert(0, self.doocs_dev.get_value())
            n_shots = int(self.ui.sb_n_shots.value())
            if len(self.peak) > n_shots:
                self.peak = self.peak[:n_shots]
                self.doocs_vals = self.doocs_vals[:n_shots]

            if self.ui.scan_tab.currentIndex() == 1:
                self.single_scatter.setData(self.peak, self.doocs_vals)
                
        if current_mode == "Peak Ampl" and self.doocs_dev is not None:
            peak_max = np.max(self.parent.data_2d[:, 0])
            self.peak.insert(0, peak_max)
            self.doocs_vals.insert(0, self.doocs_dev.get_value())
            n_shots = int(self.ui.sb_n_shots.value())
            if len(self.peak) > n_shots:
                self.peak = self.peak[:n_shots]
                self.doocs_vals = self.doocs_vals[:n_shots]

            if self.ui.scan_tab.currentIndex() == 1:
                self.single_scatter.setData(self.peak, self.doocs_vals)


    def add_corel_plot(self):
        gui_index = self.ui.get_style_name_index()
        if "standard" in self.parent.gui_styles[gui_index]:
            pg.setConfigOption('background', 'w')
            pg.setConfigOption('foreground', 'k')
            single_pen = pg.mkPen("k")
        else:
            single_pen = pg.mkPen("w")

        win = pg.GraphicsLayoutWidget()

        self.plot_cor = win.addPlot(row=0, col=0)
        self.plot_cor.setLabel('left', "A", units='au')
        self.plot_cor.setLabel('bottom', "", units='eV')

        self.plot_cor.showGrid(1, 1, 1)

        self.plot_cor.getAxis('left').enableAutoSIPrefix(enable=False)  # stop the auto unit scaling on y axes
        layout = QtGui.QGridLayout()
        self.ui.widget_correl.setLayout(layout)
        layout.addWidget(win, 0, 0)

        self.plot_cor.setAutoVisible(y=True)

        self.plot_cor.addLegend()
        pen = pg.mkPen((255, 0, 0), width=2)
        self.single_scatter = pg.ScatterPlotItem(pen=pen, name='')

        self.plot_cor.addItem(self.single_scatter)

        pen = pg.mkPen((255, 0, 0), width=2)
        # self.average = pg.PlotCurveItem(x=[], y=[], pen=pen, name='average')
        self.average_scatter = pg.ScatterPlotItem(pen=pen, name='average')

        self.plot_cor.addItem(self.average_scatter)

        pen = pg.mkPen((255, 255, 255), width=2)

        self.fit_func_scatteer = pg.PlotCurveItem(pen=pen, name='Gauss Fit')

        # self.plot1.addItem(self.fit_func)
        self.plot_cor.enableAutoRange(False)
        #self.textItem = pg.TextItem(text="", border='w', fill=(0, 0, 0))

    def add_hist_plot(self):
        gui_index = self.ui.get_style_name_index()
        if "standard" in self.parent.gui_styles[gui_index]:
            pg.setConfigOption('background', 'w')
            pg.setConfigOption('foreground', 'k')
            single_pen = pg.mkPen("k")
        else:
            single_pen = pg.mkPen("w")
        win = pg.GraphicsLayoutWidget()

        self.plot_hist = win.addPlot(row=0, col=0)
        self.plot_hist.setLabel('left', "N Events", units='')
        self.plot_hist.setLabel('bottom', "", units='a.u.')
        self.plot_hist.showGrid(1, 1, 1)
        #self.plot_hist.getAxis('left').enableAutoSIPrefix(enable=False)  # stop the auto unit scaling on y axes
        layout = QtGui.QGridLayout()
        self.ui.widget_hist.setLayout(layout)
        layout.addWidget(win, 0, 0)

        #self.plot_hist.setAutoVisible(y=True)
        #self.plot_hist.setAutoVisible(x=True)
        self.plot_hist.addLegend()
        pen = pg.mkPen((255, 0, 0), width=2)
        self.hist_item = pg.PlotDataItem(pen=pen, name='Gauss Fit') #[0,0],[0], stepMode=True, fillLevel=0, brush=(0,0,255,150), name='single')

        self.plot_hist.addItem(self.hist_item)



        #pen = pg.mkPen((255, 255, 255), width=2)

        #self.fit_func_hist = pg.PlotCurveItem(pen=pen, name='Gauss Fit')

        # self.plot1.addItem(self.fit_func)
        #self.plot_hist.enableAutoRange(False)
        #self.textItem = pg.TextItem(text="", border='w', fill=(0, 0, 0))