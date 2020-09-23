import numpy as np
import pyqtgraph as pg
from threading import Thread, Event
from PyQt5 import QtGui, QtCore
import time
from mint.opt_objects import Device
from scipy import ndimage
from matplotlib import cm


class Correl2DInterface:
    """
    Main class for 2D correlation
    """
    def __init__(self, parent):
        self.parent = parent
        self.ui = self.parent.ui
        self.mi = self.parent.mi

        # self.add_corel2D_plot() #TMP~~~~~~~~~~~~~~~~~
        
        # self.add_hist_plot()
        #self.plot1.scene().sigMouseMoved.connect(self.mouseMoved)
        # self.ui.cb_corel_spect.addItem("Peak Pos")
        # self.ui.cb_corel_spect.addItem("Peak Ampl")
        self.doocs_dev = None
        # self.doocs_dev_hist = None
        self.get_device()
        self.ui.le_doocs_ch_cor2d.editingFinished.connect(self.get_device)
        
        self.plot_timer = pg.QtCore.QTimer()
        self.plot_timer.timeout.connect(self.plot_correl)
        self.plot_timer.start(100)

        # self.plot_timer_hist = pg.QtCore.QTimer()
        # self.plot_timer_hist.timeout.connect(self.plot_histogram)
        # self.plot_timer_hist.start(100)

        self.phen = []
        self.spec_hist = []
        self.doocs_vals_hist = []
        
        self.spec_binned = []
        self.doocs_bins = []
        
        self.ui.sb_corr_2d_reset.clicked.connect(self.reset)

        # self.ui.pb_start_scan.clicked.connect(self.start_stop_scan)
        # self.ui.pb_check_range.clicked.connect(self.check_range)
        # self.ui.pb_show_map.clicked.connect(self.show_hide_map)
        
        self.add_image_widget()

    def get_device(self):
        if self.ui.is_le_addr_ok(self.ui.le_doocs_ch_cor2d):
            eid = self.ui.le_doocs_ch_cor2d.text()
            self.doocs_dev = Device(eid=eid)
            self.doocs_dev.mi = self.mi
        else:
            self.doocs_dev = None
            
    
    def sort_and_bin(self):
        n_phens = self.parent.spectrum_event.size
        try:
            sbin = float(self.ui.sb_corr2d_binning.text()) #bin size
        except ValueError:
            sbin = 0
        
        if sbin==0:
            sbin=1e10
        
        min_val = sbin * (int(min(self.doocs_vals_hist) / sbin)-1)
        max_val = max(self.doocs_vals_hist)
        print('min_val', min_val)
        print('max_val', max_val)
        print('sbin', sbin)
        self.doocs_bins = np.arange(min_val, max_val, sbin)
        
        spec = np.array(self.spec_hist)
        print('shape of spec array', spec.shape)
        # print(self.doocs_bins)
        
        
        bin_dest_idx = np.digitize(self.doocs_vals_hist, self.doocs_bins) - 1
        self.spec_binned = np.zeros((len(self.doocs_bins), n_phens))
        
        for i in np.unique(bin_dest_idx):
            idx = np.where(i == bin_dest_idx)[0]
            # print('sorting', i, idx)
            if len(idx) > 1:
                self.spec_binned[i, :] = np.mean(spec[idx, :], axis=0)
                # print(spectra[:, idx].shape)
            elif len(idx) == 1:
                # print('singe', i, idx)
                self.spec_binned[i, :] = spec[idx[0], :]
            else:
                pass
                
    def reset(self):
        self.spec_hist = []
        self.doocs_vals_hist = []
        

    def plot_correl(self):

        # current_mode = self.ui.cb_corel_spect.currentText()
        if self.ui.pb_start.text() == "Start" or not self.ui.sb_corr_2d_run.isChecked():
            return
        # if current_mode == "Peak Pos" and self.doocs_dev is not None:
        
        
        
        self.phen = self.parent.x_axis
        
        self.spec_hist.append(self.parent.spectrum_event)
        
        #self.doocs_dev is None and
        if self.parent.ui.combo_hirex.currentText() == "DUMMY HIREX":
        
            dummy_val = np.sin(time.time()/5)*7.565432
            self.doocs_vals_hist.append(dummy_val) #fix
            print('dummy hirex value', dummy_val)
            self.doocs_address_label = 'dummy label'
        else:
            self.doocs_vals_hist.append(self.doocs_dev.get_value())
            self.doocs_address_label = self.ui.le_doocs_ch_cor2d.text()
        
        n_shots = int(self.ui.sb_n_shots_2d.value())
        
        
        
        if len(self.spec_hist) > n_shots:
            self.spec_hist = self.spec_hist[:n_shots]
            self.doocs_vals_hist = self.doocs_vals_hist[:n_shots]
            
        print(self.doocs_vals_hist[-1])
        # print(len(self.spec_hist), len(self.doocs_vals_hist))
        # print(min(self.doocs_vals_hist), max(self.doocs_vals_hist), self.ui.sb_corr2d_binning.value())
        
        
        
        self.sort_and_bin()
        
        print('final shape', self.spec_binned.shape)
        
        

        # if self.ui.scan_tab.currentIndex() == 1:
            # self.single_scatter.setData(self.peak, self.doocs_vals)
        

        # peak_max = np.max(self.parent.data_2d[:, 0])
        # self.peak.insert(0, peak_max)
        # self.doocs_vals.insert(0, self.doocs_dev.get_value())
        # n_shots = int(self.ui.sb_av_nbunch.value())
        # if len(self.peak) > n_shots:
            # self.peak = self.peak[:n_shots]
            # self.doocs_vals = self.doocs_vals[:n_shots]
            
        # print(dir(self.ui.scan_tab.currentWidget()))
        
        
        
        
        if self.ui.scan_tab.currentIndex() == 2:
        
            # scale_xaxis = (self.phen[-1] - self.phen[0]) / len(self.phen)
            # shift_xaxis = self.phen[0] / scale_xaxis
            # self.single_scatter.setData(self.peak, self.doocs_vals)
            self.img.setImage(self.spec_binned)
            # self.img.scale(scale_xaxis, 1)
            
            
            
    def add_image_widget(self):
        win = pg.GraphicsLayoutWidget()
        layout = QtGui.QGridLayout()
        self.ui.widget_corr2d.setLayout(layout)
        layout.addWidget(win)

        self.img_plot = win.addPlot()
        self.add_image_item()
        
    def add_image_item(self):
        self.img_plot.clear()

        # self.img_plot.setLabel('E_photon', "keV", units='')
        # self.img_plot.setLabel('bottom', 'bottom', units='')

        self.img = pg.ImageItem()

        self.img_plot.addItem(self.img)

        colormap = cm.get_cmap('viridis') #"nipy_spectral")  # cm.get_cmap("CMRmap")
        colormap._init()
        lut = (colormap._lut * 255).view(np.ndarray)  # Convert matplotlib colormap from 0-1 to 0 -255 for Qt

        # Apply the colormap
        self.img.setLookupTable(lut)


    def add_corel2D_plot(self):
        gui_index = self.ui.get_style_name_index()
        if "standard" in self.parent.gui_styles[gui_index]:
            pg.setConfigOption('background', 'w')
            pg.setConfigOption('foreground', 'k')
            single_pen = pg.mkPen("k")
        else:
            single_pen = pg.mkPen("w")

        win = pg.GraphicsLayoutWidget()

        self.plot_cor = win.addPlot(row=0, col=0)
        self.plot_cor.setLabel('left', "E_ph", units='keV')
        self.plot_cor.setLabel('bottom', self.doocs_address_label, units='')

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



        #pen = pg.mkPen((255, 255, 255), width=2)

        #self.fit_func_hist = pg.PlotCurveItem(pen=pen, name='Gauss Fit')

        # self.plot1.addItem(self.fit_func)
        #self.plot_hist.enableAutoRange(False)
        #self.textItem = pg.TextItem(text="", border='w', fill=(0, 0, 0))