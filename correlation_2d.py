import numpy as np
import pyqtgraph as pg
from threading import Thread, Event
from PyQt5 import QtGui, QtCore
import time
from mint.opt_objects import Device
from scipy import ndimage
from matplotlib import cm

def find_nearest_idx(array, value):
    idx = np.abs(array - value).argmin()
    return idx

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
        
        self.calc_timer = pg.QtCore.QTimer()
        self.calc_timer.timeout.connect(self.calc_correl)
        self.calc_timer.start(100)
        
        self.plot_timer = pg.QtCore.QTimer()
        self.plot_timer.timeout.connect(self.plot_correl)
        self.plot_timer.timeout.connect(self.plot_hist_event)
        self.plot_timer.timeout.connect(self.plot_Ipk_event)
        self.plot_timer.timeout.connect(self.plot_Isum_event)
        self.plot_timer.start(200)
        
        # self.plot_timer_hist_event = pg.QtCore.QTimer()
        # self.plot_timer_hist_event.timeout.connect(self.plot_hist_event)
        # self.plot_timer_hist_event.start(100)
        
        # self.plot_timer_Ipk_event = pg.QtCore.QTimer()
        # self.plot_timer_Ipk_event.timeout.connect(self.plot_Ipk_event)
        # self.plot_timer_Ipk_event.start(100)
        
        # self.plot_timer_Isum_event = pg.QtCore.QTimer()
        # self.plot_timer_Isum_event.timeout.connect(self.plot_Isum_event)
        # self.plot_timer_Isum_event.start(100)

        # self.plot_timer_hist = pg.QtCore.QTimer()
        # self.plot_timer_hist.timeout.connect(self.plot_histogram)
        # self.plot_timer_hist.start(100)

        self.phen = []
        self.spec_hist = []
        self.doocs_vals_hist = []
        self.transmission_vals_hist = []
        self.cross_callibration_vals_hist = []
        
        self.doocs_old_label = ''
        
        self.spec_binned = []
        self.doocs_bins = []
        self.doocs_event_counts = []
        self.doocs_vals_hist_lagged = []
        
        self.event_counter = 0
        
        self.doocs_address_label = self.ui.le_doocs_ch_cor2d.text()
        
        # self.channel_timer = pg.QtCore.QTimer()
        # if self.doocs_address_label != self.ui.le_doocs_ch_cor2d.text():
            # self.reset()
            # self.doocs_address_label = self.ui.le_doocs_ch_cor2d.text()
        # self.channel_timer.start(100)
        
        
        self.ui.sb_corr_2d_reset.clicked.connect(self.reset)

        # self.ui.pb_start_scan.clicked.connect(self.start_stop_scan)
        # self.ui.pb_check_range.clicked.connect(self.check_range)
        # self.ui.pb_show_map.clicked.connect(self.show_hide_map)
        
        self.doocs_address_label = '' #костыль
        self.add_corr2d_image_widget()
        self.add_hist_event_widget()
        self.add_corrIpk_widget()
        self.add_corrIsum_widget()
        
        self.ui.actionSave_Corelation.triggered.connect(self.save_corr2d_data_as)

    def stop_timers(self):
        self.plot_timer.stop()
        # self.plot_timer_hist_event.stop()
        # self.plot_timer_Ipk_event.stop()
        # self.plot_timer_Isum_event.stop()

    def get_device(self):
        if self.ui.is_le_addr_ok(self.ui.le_doocs_ch_cor2d):
            eid = self.ui.le_doocs_ch_cor2d.text()
            self.doocs_dev = Device(eid=eid)
            self.doocs_dev.mi = self.mi
        else:
            self.doocs_dev = None
            
    
    def sort_and_bin(self):
        
        
        try:
            bin_doocs = float(self.ui.sb_corr2d_binning.text()) #bin size
        except ValueError:
            bin_doocs = 0
            
        try:
            phen_min = self.ui.sb_emin.value()/1000
        except ValueError:
            phen_min = -np.inf
            
        self.phen_orig = self.parent.x_axis_disp
            
        try:
            phen_max = self.ui.sb_emax.value()/1000
        except ValueError:
            phen_max = np.inf
            
        # print('phen_max',phen_max)
        # print('phen_min',phen_min)
        
        d2, d1 = 0, 0
        
        if phen_max > phen_min:
            d1 = find_nearest_idx(self.phen_orig/1000, phen_min)
            d2 = find_nearest_idx(self.phen_orig/1000, phen_max)
        # else:
        if d2 <= d1:
            d1 = 0
            d2 = len(self.phen_orig)
        
        
        
        #print('d1, d2', d1, d2)
        self.phen = self.phen_orig[d1:d2]
        n_phens = len(self.phen)
        
        #print('self.phen_orig',len(self.phen_orig))
        #print('self.phen',len(self.phen))
        
        if bin_doocs==0:
            bin_doocs=1e10
            
        try:
            self.n_lag = int(self.ui.sb_corr2d_lag.value()) #lag size
        except ValueError:
            self.n_lag = 0
        
        #print('len(self.doocs_vals_hist)', len(self.doocs_vals_hist))
        if len(self.doocs_vals_hist) > abs(self.n_lag)+5:
            if self.n_lag >= 0:
                self.doocs_vals_hist_lagged = self.doocs_vals_hist[:len(self.doocs_vals_hist)-self.n_lag]
                spec_lagged = np.array(self.spec_hist)[self.n_lag:, :]
            else:
                self.doocs_vals_hist_lagged = self.doocs_vals_hist[abs(self.n_lag):]
                spec_lagged = np.array(self.spec_hist)[:len(self.doocs_vals_hist)-abs(self.n_lag), :]
                
        else:
            self.doocs_vals_hist_lagged = self.doocs_vals_hist
            spec_lagged = np.array(self.spec_hist) * np.array(self.cross_callibration_vals_hist)[:, None] / np.array(self.transmission_vals_hist)[:,None] # TODO: untested!!!
            #spec_lagged = np.array(self.spec_hist)
        
        
        
        # spec = np.array(self.spec_hist)
        # min_val = bin_doocs * (int(min(self.doocs_vals_hist) / bin_doocs))
        # max_val = max(self.doocs_vals_hist)
        min_val = bin_doocs * (int(np.nanmin(self.doocs_vals_hist_lagged) / bin_doocs)) #ensures the minimum value is integer of bin width and figure does not jitter
        max_val = max(self.doocs_vals_hist_lagged) + bin_doocs * 1.01
        
        if max_val - min_val <= bin_doocs:
            max_val = min_val + 1.01 * bin_doocs #ensures there is at least one bin (two bin values)
        
        # if min_val == max_val:
            # max_val = 1.001 * min_val #ensures there is at least one bin
        
        #print('min_DOOCS_val', min_val)
        #print('max_DOOCS_val', max_val)
        # print('bin_doocs', bin_doocs)
        self.doocs_bins = np.arange(min_val, max_val, bin_doocs)
        # print('shape of created doocs_bins', self.doocs_bins.shape)
        
        self.doocs_event_counts, _ = np.histogram(self.doocs_vals_hist_lagged, bins=self.doocs_bins)
        
        #print('len spec_hist',len(self.spec_hist))
        #print('len doocs_vals_hist',len(self.doocs_vals_hist))
        
        #print('shape of spec_lagged array', spec_lagged.shape)
        #print('shape of doocs_bins', self.doocs_bins.shape)
        #print('doocs_bins',len(self.doocs_bins), self.doocs_bins)
        
        self.bin_dest_idx = np.digitize(self.doocs_vals_hist_lagged, self.doocs_bins)-1
        self.spec_binned = np.zeros((len(self.doocs_bins)-1, n_phens))
        
        #print('self.bin_dest_idx', self.bin_dest_idx)
        #print('self.spec_binned', len(self.spec_binned), self.spec_binned)
        #print('spec_lagged', len(spec_lagged), spec_lagged)
        #print('self.doocs_vals_hist_lagged', len(self.doocs_vals_hist_lagged), self.doocs_vals_hist_lagged)
        
        for i in np.unique(self.bin_dest_idx):
            idx = np.where(i == self.bin_dest_idx)[0]
            # print('sorting', i, idx)
            if len(idx) > 1:
                self.spec_binned[i, :] = np.mean(spec_lagged[idx, d1:d2], axis=0)
                #print('multiple', i, idx, len(self.spec_binned))
            elif len(idx) == 1:
                #print('singe', i, idx, len(self.spec_binned))
                # if len(self.spec_binned) != 0:
                #print('self.spec_binned[i, :]', self.spec_binned[i, :])
                #print('spec_lagged[idx[0], d1:d2]', spec_lagged[idx[0], d1:d2])
                self.spec_binned[i, :] = spec_lagged[idx[0], d1:d2]
                # else:
                    # self.spec_binned = np.array([spec_lagged[idx[0], :]])
            else:
                pass
        
        #print('self.spec_binned', self.spec_binned)
        #print('self.doocs_bins', self.doocs_bins)
        # if 
        
        # self.doocs_event_counts = np.unique(self.bin_dest_idx, return_counts=1)[1]
        
        
    def reset(self):
        self.spec_hist = []
        self.doocs_vals_hist = []
        self.img_hist.clear()
        self.img_corr2d.clear()
        self.img_Ipk.clear()
        self.img_Isum.clear()
        self.doocs_address_label = self.ui.le_doocs_ch_cor2d.text()
        self.transmission_vals_hist = []
        self.cross_callibration_vals_hist = []
        self.doocs_bins = []
        self.event_counter = 0
        
    def calc_correl(self):
        if self.ui.pb_start.text() == "Start" or not self.ui.sb_corr_2d_run.isChecked() or self.parent.spectrum_event_disp is None:
            return
            
        # print('DOOCS LABEL ',self.doocs_address_label)
        if self.ui.le_doocs_ch_cor2d.text() != self.doocs_address_label:
            self.reset()
            return
        
        self.event_counter += 1
        
        # print('min_self.phen_val', min(self.phen))
        # print('max_self.phen_val', max(self.phen))
        
        if self.parent.energy_axis_thread.trigger:
            print('energy_axis_thread.trigger')
            self.reset()
            
        if len(self.parent.ave_spectrum) < 3:
            print('self.parent.ave_spectrum < 3')
            self.reset()
        
        n_shots = int(self.ui.sb_n_shots_max.value())
        if len(self.spec_hist) > n_shots: #add lag value
            self.spec_hist = self.spec_hist[-n_shots:]
            self.doocs_vals_hist = self.doocs_vals_hist[-n_shots:]
        
        if len(self.spec_hist) >0:
            if len(self.spec_hist[0]) != len(self.parent.spectrum_event_disp):
                self.reset()
        self.spec_hist.append(self.parent.spectrum_event_disp)
        
        
        #print(self.doocs_address_label)
        
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
            
        self.transmission_vals_hist.append(self.parent.transmission_value)
        self.cross_callibration_vals_hist.append(self.parent.calib_energy_coef)
        
        self.doocs_old_label = self.doocs_address_label
        
        if self.ui.scan_tab.currentIndex() == 2:
            self.sort_and_bin()

        # print(self.doocs_vals_hist[-1])
    def plot_correl(self):
        if self.ui.pb_start.text() == "Start" or not self.ui.sb_corr_2d_run.isChecked() or self.parent.spectrum_event_disp is None or self.ui.scan_tab.currentIndex() != 2:
            return
        
        try:
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
            self.img_corr2d.setLabel('bottom', self.doocs_address_label, units='_')
        except:
            pass     
        
        

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
        # self.img_hist.setLabel('left', "N Events", units='')
        # self.img_hist.showGrid(1, 1, 1)
        self.img_hist.setLabel('bottom', self.doocs_address_label, units='_')
        
    def add_corr2d_image_item(self):
        self.img_corr2d.clear()

        self.img_corr2d.setLabel('left', "E_ph", units='eV')
        self.img_corr2d.setLabel('bottom', self.doocs_address_label, units='_')

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
                self.img_hist.plot(self.doocs_bins, self.doocs_event_counts, stepMode=True,  fillLevel=0,  brush=(100,100,100,150), clear=True)
                self.img_hist.setLabel('bottom', self.doocs_address_label, units=' ')
                self.img_hist.setLabel('left','{} events'.format(len(self.doocs_vals_hist_lagged)))
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
        # self.img_Ipk.showGrid(1, 1, 1)
        self.img_Ipk.setLabel('bottom', self.doocs_address_label, units='_')
        
    def plot_Ipk_event(self):
        if self.ui.pb_start.text() == "Start" or not self.ui.sb_corr_2d_run.isChecked():
            return
        if self.ui.scan_tab.currentIndex() == 2:
            self.img_Ipk.clear()
            if len(self.doocs_bins) > 1:
                
                bin_step = self.doocs_bins[1] - self.doocs_bins[0]
                interbin_values = self.doocs_bins[:-1] + bin_step/2
                
                # print('plot_Ipk', len(interbin_values), len(self.spec_binned))
                pen=pg.mkPen(color=(255, 0, 0), width=3)
                self.img_Ipk.plot(interbin_values, np.amax(self.spec_binned,axis=1), stepMode=False, pen=pen)#,  fillLevel=0,  brush=(0,0,255,150), clear=True)
                self.img_Ipk.setLabel('bottom', self.doocs_address_label, units=' ')
                # self.img_Ipk.setLimits(yMin=0)
        
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
        self.img_Isum.setLabel('left', "Isum", units='')
        # self.img_Isum.showGrid(1, 1, 1)
        self.img_Isum.setLabel('bottom', self.doocs_address_label, units=' ')
        
    def plot_Isum_event(self):
        if self.ui.pb_start.text() == "Start" or not self.ui.sb_corr_2d_run.isChecked():
            return
        if self.ui.scan_tab.currentIndex() == 2:
            self.img_Isum.clear()
            if len(self.doocs_bins) > 1:
                
                bin_step = self.doocs_bins[1] - self.doocs_bins[0]
                interbin_values = self.doocs_bins[:-1] + bin_step/2
                
                # print('plot_Ipk', len(interbin_values), len(self.spec_binned))
                pen=pg.mkPen(color=(0, 0, 255), width=3)
                #self.img_Isum.plot(interbin_values, np.amax(self.spec_binned,axis=1)/np.sum(self.spec_binned,axis=1), stepMode=False, pen=pen)#,  fillLevel=0,  brush=(0,0,255,150), clear=True)
                self.img_Isum.plot(interbin_values, np.sum(self.spec_binned,axis=1), stepMode=False, pen=pen)#,  fillLevel=0,  brush=(0,0,255,150), clear=True)
                self.img_Isum.setLabel('bottom', self.doocs_address_label, units=' ')
        
    def save_corr2d_data_as(self):
        filename = QtGui.QFileDialog.getSaveFileName(self.parent, 'Save Correlation&Spectrum Data',
                                                     self.parent.data_dir, "txt (*.npz)", None,
                                                     QtGui.QFileDialog.DontUseNativeDialog
                                                     )[0]
        np.savez(filename, phen_scale=self.phen, spec_hist=self.spec_hist, doocs_vals_hist=self.doocs_vals_hist, corr2d=self.spec_binned, doocs_scale = self.doocs_bins, doocs_channel=self.doocs_address_label)