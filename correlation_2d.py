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
        
        # self.plot_timer_hist_event = pg.QtCore.QTimer()
        # self.plot_timer_hist_event.timeout.connect(self.plot_hist_event)
        # self.plot_timer_hist_event.start(100)
        
        # self.plot_timer_Ipk_event = pg.QtCore.QTimer()
        # self.plot_timer_Ipk_event.timeout.connect(self.plot_Ipk_event)
        # self.plot_timer_Ipk_event.start(100)

        # self.plot_timer_hist = pg.QtCore.QTimer()
        # self.plot_timer_hist.timeout.connect(self.plot_histogram)
        # self.plot_timer_hist.start(100)

        self.phen = []
        self.spec_hist = []
        self.doocs_vals_hist = []
        
        self.spec_binned = []
        self.doocs_bins = []
        self.doocs_event_counts = []
        self.doocs_vals_hist_lagged = []
        
        # self.doocs_address_label = self.ui.le_doocs_ch_cor2d.text()
        
        # self.channel_timer = pg.QtCore.QTimer()
        # doocs_dev_eid
        # if self.doocs_address_label != self.ui.le_doocs_ch_cor2d.text():
            # self.reset()
            # self.doocs_address_label = self.ui.le_doocs_ch_cor2d.text()
        # self.channel_timer.start(100)
        
        
        self.ui.sb_corr_2d_reset.clicked.connect(self.reset)

        # self.ui.pb_start_scan.clicked.connect(self.start_stop_scan)
        # self.ui.pb_check_range.clicked.connect(self.check_range)
        # self.ui.pb_show_map.clicked.connect(self.show_hide_map)
        
        self.doocs_dev_eid = '' #костыль
        self.add_corr2d_image_widget()
        self.add_hist_event_widget()
        self.add_corrIpk_widget()
        self.add_corrIsum_widget()
        
        self.ui.actionSave_Corelation.triggered.connect(self.save_corr2d_data_as)

    def get_device(self):
        if self.ui.is_le_addr_ok(self.ui.le_doocs_ch_cor2d):
            self.doocs_dev_eid = self.ui.le_doocs_ch_cor2d.text()
            self.doocs_dev = Device(eid=self.doocs_dev_eid)
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
            
        try:
            self.n_lag = int(self.ui.sb_corr2d_lag.text()) #bin size
        except ValueError:
            self.n_lag = 0
        
        
        if len(self.doocs_vals_hist) > abs(self.n_lag)+5:
            if self.n_lag >= 0:
                self.doocs_vals_hist_lagged = self.doocs_vals_hist[:len(self.doocs_vals_hist)-self.n_lag]
                spec_lagged = np.array(self.spec_hist)[self.n_lag:, :]
            else:
                self.doocs_vals_hist_lagged = self.doocs_vals_hist[abs(self.n_lag):]
                spec_lagged = np.array(self.spec_hist)[:len(self.doocs_vals_hist)-abs(self.n_lag), :]
                
        else:
            self.doocs_vals_hist_lagged = self.doocs_vals_hist
            spec_lagged = np.array(self.spec_hist)
        
        
        
        # spec = np.array(self.spec_hist)
        # min_val = sbin * (int(min(self.doocs_vals_hist) / sbin))
        # max_val = max(self.doocs_vals_hist)
        min_val = sbin * (int(min(self.doocs_vals_hist_lagged) / sbin)) #ensures the minimum value is integer of bin width and figure does not jitter
        max_val = max(self.doocs_vals_hist_lagged) + sbin
        
        if min_val == max_val:
            max_val = 1.001 * min_val #ensures there is at least one bin
        
        print('min_DOOCS_val', min_val)
        print('max_DOOCS_val', max_val)
        print('sbin', sbin)
        self.doocs_bins = np.arange(min_val, max_val, sbin)
        
        self.doocs_event_counts, self.doocs_bins = np.histogram(self.doocs_vals_hist_lagged, bins=self.doocs_bins)
        
        print('len spec_hist',len(self.spec_hist))
        print('len doocs_vals_hist',len(self.doocs_vals_hist))
        
        print('shape of spec_lagged array', spec_lagged.shape)
        print('shape of doocs_bins', self.doocs_bins.shape)
        print('doocs_bins',self.doocs_bins)
        
        self.bin_dest_idx = np.digitize(self.doocs_vals_hist_lagged, self.doocs_bins)-1
        self.spec_binned = np.zeros((len(self.doocs_bins)-1, n_phens))
        # if len(self.spec_binned) == 0:
            # print('len(self.spec_binned) == 0, replacing')
            # self.spec_binned = spec_lagged
        # else:
        
        print('self.bin_dest_idx', self.bin_dest_idx)
        print('self.spec_binned', self.spec_binned)
        
        for i in np.unique(self.bin_dest_idx):
            idx = np.where(i == self.bin_dest_idx)[0]
            
            if len(idx) > 1:
                self.spec_binned[i, :] = np.mean(spec_lagged[idx, :], axis=0)
                print('multiple', i, idx)
            elif len(idx) == 1:
                print('singe', i, idx)
                self.spec_binned[i, :] = spec_lagged[idx[0], :]
            else:
                pass
        
        # self.doocs_event_counts = np.unique(self.bin_dest_idx, return_counts=1)[1]
        
                
    def reset(self):
        self.spec_hist = []
        self.doocs_vals_hist = []
        self.img_hist.clear()
        self.img_corr2d.clear()
        
    def plot_correl(self):

        if self.ui.pb_start.text() == "Start" or not self.ui.sb_corr_2d_run.isChecked() or self.parent.spectrum_event is None:
            return
        
        self.phen = self.parent.x_axis
        # print('min_self.phen_val', min(self.phen))
        # print('max_self.phen_val', max(self.phen))
        
        self.spec_hist.append(self.parent.spectrum_event)
        
        
        #print(self.doocs_dev_eid)
        doocs_dev_eid_old = self.doocs_dev_eid
        self.get_device()
        if doocs_dev_eid_old != self.doocs_dev_eid:
            self.reset()
        
        
        if self.doocs_dev_eid == 'event':
            self.doocs_vals_hist.append(len(self.spec_hist))
                
        if self.parent.ui.combo_hirex.currentText() == "DUMMY HIREX":
            dummy_val = np.sin(time.time()/10)*7.565432 + 25
            if self.doocs_dev_eid != 'event':
                self.doocs_vals_hist.append(dummy_val) #fix
                self.doocs_dev_eid = 'dummy label'
        else:
            self.doocs_vals_hist.append(self.doocs_dev.get_value())
            
        
        n_shots = int(self.ui.sb_n_shots_2d.value())
        
        
        if len(self.spec_hist) > n_shots: #add lag value
            self.spec_hist = self.spec_hist[-n_shots:]
            self.doocs_vals_hist = self.doocs_vals_hist[-n_shots:]
        # print(self.doocs_vals_hist[-1])
        
        if self.ui.scan_tab.currentIndex() == 2:
            self.sort_and_bin()
            scale_yaxis = (self.phen[-1] - self.phen[0]) / len(self.phen)
            translate_yaxis = self.phen[0] / scale_yaxis
            
            scale_xaxis = (max(self.doocs_bins) - min(self.doocs_bins)) / len(self.doocs_bins)
            translate_xaxis = min(self.doocs_bins) / scale_xaxis
            # self.single_scatter.setData(self.peak, self.doocs_vals)
            
            self.add_corr2d_image_item()
            self.img.setImage(self.spec_binned)
            # print('spec binned',len(self.spec_binned))
            # print('doocs_bins',len(self.doocs_bins))
            #elegant but maybe wrong
            # rect = QtCore.QRect(min(self.doocs_bins), min(self.phen), max(self.doocs_bins)-min(self.doocs_bins), max(self.phen)-min(self.phen))
            # self.img.setRect(rect)
            self.img.scale(scale_xaxis, scale_yaxis)
            self.img.translate(translate_xaxis, translate_yaxis)
        

    def add_corr2d_image_widget(self):
        win = pg.GraphicsLayoutWidget()
        layout = QtGui.QGridLayout()
        self.ui.widget_corr2d.setLayout(layout)
        layout.addWidget(win)

        self.img_corr2d = win.addPlot()
        self.add_corr2d_image_item()
        
    def add_hist_event_widget(self):
        gui_index = self.ui.get_style_name_index()
        if "standard" in self.parent.gui_styles[gui_index]:
            pg.setConfigOption('background', 'w')
            pg.setConfigOption('foreground', 'k')
            single_pen = pg.mkPen("k")
        else:
            single_pen = pg.mkPen("w")
        win = pg.GraphicsLayoutWidget()
        
        layout = QtGui.QGridLayout()
        self.ui.widget_corr_event.setLayout(layout)
        layout.addWidget(win)

        self.img_hist = win.addPlot()
        self.img_hist.setLabel('left', "N Events", units='')
        self.img_hist.showGrid(1, 1, 1)
        self.img_hist.setLabel('bottom', self.doocs_dev_eid, units='_')
        
    def add_corr2d_image_item(self):
        self.img_corr2d.clear()

        self.img_corr2d.setLabel('left', "E_ph", units='eV')
        self.img_corr2d.setLabel('bottom', self.doocs_dev_eid, units='_')

        self.img = pg.ImageItem()
        self.img_corr2d.addItem(self.img)

        colormap = cm.get_cmap('viridis') #"nipy_spectral")  # cm.get_cmap("CMRmap")
        colormap._init()
        lut = (colormap._lut * 255).view(np.ndarray)  # Convert matplotlib colormap from 0-1 to 0 -255 for Qt
        # Apply the colormap
        self.img.setLookupTable(lut)
        
    def plot_hist_event(self):
    
        if self.ui.pb_start.text() == "Start" or not self.ui.sb_corr_2d_run.isChecked():
            return
        
        # if len(self.doocs_vals_hist_lagged) < 2:
            # return
            
        if self.ui.scan_tab.currentIndex() == 2:
            self.img_hist.clear()
            
            # print('bins', self.doocs_bins)
            # print('events', self.doocs_event_counts)
            
            if len(self.doocs_bins) > 1:
                self.img_hist.plot(self.doocs_bins, self.doocs_event_counts, stepMode=True,  fillLevel=0,  brush=(0,0,255,150), clear=True)
        # self.img = pg.ImageItem()
        # self.img_hist.addItem(self.img)

        # colormap = cm.get_cmap('viridis') #"nipy_spectral")  # cm.get_cmap("CMRmap")
        # colormap._init()
        # lut = (colormap._lut * 255).view(np.ndarray)  # Convert matplotlib colormap from 0-1 to 0 -255 for Qt
        # Apply the colormap
    
            
    def add_corrIpk_widget(self):
        gui_index = self.ui.get_style_name_index()
        if "standard" in self.parent.gui_styles[gui_index]:
            pg.setConfigOption('background', 'w')
            pg.setConfigOption('foreground', 'k')
            single_pen = pg.mkPen("k")
        else:
            single_pen = pg.mkPen("w")
        win = pg.GraphicsLayoutWidget()
        
        layout = QtGui.QGridLayout()
        self.ui.widget_corr_Ipeak.setLayout(layout)
        layout.addWidget(win)

        self.img_Ipk = win.addPlot()
        self.img_Ipk.setLabel('left', "I (peak)", units='')
        self.img_Ipk.showGrid(1, 1, 1)
        self.img_Ipk.setLabel('bottom', self.doocs_dev_eid, units='_')
        
    def plot_Ipk_event(self):
    
        if self.ui.pb_start.text() == "Start" or not self.ui.sb_corr_2d_run.isChecked():
            return
        
        # if len(self.doocs_vals_hist_lagged) < 2:
            # return
            
        if self.ui.scan_tab.currentIndex() == 2:
            self.img_Ipk.clear()
            
            # print('bins', self.doocs_bins)
            # print('events', self.doocs_event_counts)
            
            if len(self.doocs_bins) > 1:
                bin_step = self.doocs_bins[1] - self.doocs_bins[0]
                interbin_values = self.doocs_bins[1:] - bin_step/2
                self.img_Ipk.plot(interbin_values, np.amax(self.spec_binned,axis=1))#, stepMode=False,  fillLevel=0,  brush=(0,0,255,150), clear=True)
        
    def add_corrIsum_widget(self):
        gui_index = self.ui.get_style_name_index()
        if "standard" in self.parent.gui_styles[gui_index]:
            pg.setConfigOption('background', 'w')
            pg.setConfigOption('foreground', 'k')
            single_pen = pg.mkPen("k")
        else:
            single_pen = pg.mkPen("w")
        win = pg.GraphicsLayoutWidget()
        
        layout = QtGui.QGridLayout()
        self.ui.widget_corr_Isum.setLayout(layout)
        layout.addWidget(win)

        self.img_Isum = win.addPlot()
        self.img_Isum.setLabel('left', "I (sum)", units='')
        self.img_Isum.showGrid(1, 1, 1)
        self.img_Isum.setLabel('bottom', self.doocs_dev_eid, units='_')
        
    def save_corr2d_data_as(self):
        filename = QtGui.QFileDialog.getSaveFileName(self.parent, 'Save Correlation&Spectrum Data',
                                                     self.parent.data_dir, "txt (*.npz)", None,
                                                     QtGui.QFileDialog.DontUseNativeDialog
                                                     )[0]
        np.savez(filename, phen_scale=self.phen, spec_hist=self.spec_hist, doocs_vals_hist=self.doocs_vals_hist, corr2d=self.spec_binned, doocs_scale = self.doocs_bins, doocs_channel=self.doocs_dev_eid)