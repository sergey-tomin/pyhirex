import numpy as np
import pyqtgraph as pg
from threading import Thread, Event
from PyQt5 import QtGui, QtCore
import time
from mint.opt_objects import Device
from scipy import ndimage
from matplotlib import cm
from opt_lib import *

# from opt_lib import SpectrumArray

def find_nearest_idx(array, value):
    idx = np.abs(array - value).argmin()
    return idx

class AnalysisInterface:
    """
    Main class for 2D correlation
    """
    def __init__(self, parent):
        self.parent = parent
        self.ui = self.parent.ui
        self.mi = self.parent.mi
        
        self.spar = SpectrumArray()
        self.g2fit = FitResult()
        self.g2fit.omega=np.array([0])
        
        self.acquire_timer = pg.QtCore.QTimer()
        self.acquire_timer.timeout.connect(self.arange_spectra)
        self.acquire_timer.start(100)
        
        
        self.plot_timer = pg.QtCore.QTimer()
        self.plot_timer.timeout.connect(self.plot_spec)
        self.plot_timer.timeout.connect(self.plot_hist_full)
        self.plot_timer.timeout.connect(self.plot_hist_peak)
        self.plot_timer.start(1000)
        
        self.add_spec_widget()
        self.add_hist_full_widget()
        self.add_hist_peak_widget()
        self.add_durr_widget()
        
        self.ui.analysis_resetbutton.clicked.connect(self.reset_spectra)
        
        self.ui.analysis_correlate_button.clicked.connect(self.correlate)
    
    def stop_timers(self):
        self.acquire_timer.stop()
        self.plot_timer.stop()
        
    # def reset(self):
        # self.spar = SpectrumArray()
        
    # def add_sum_hist_event_widget(self):
        # gui_index = self.ui.get_style_name_index()
        # if "standard" in self.parent.gui_styles[gui_index]:
            # pg.setConfigOption('background', 'w')
            # pg.setConfigOption('foreground', 'k')
            # single_pen = pg.mkPen("k")
        # else:
            # single_pen = pg.mkPen("w")
        # win = pg.GraphicsLayoutWidget()
        
        # layout = QtGui.QGridLayout()
        # self.ui.widget_corr_event____________replace.setLayout(layout)
        # layout.addWidget(win)

        # self.img_sum_hist = win.addPlot()
        # self.img_sum_hist.setLabel('left', "N Events", units='')
        # # self.img_hist.showGrid(1, 1, 1)
        # self.img_sum_hist.setLabel('bottom', self.doocs_address_label, units='_')
        
        
    # def plot_sum_hist_event(self):
    
        # if self.ui.pb_start.text() == "Start" or not self.ui.sb_corr_2d_run.isChecked():
            # return
        
        # # if len(self.doocs_vals_hist_lagged) < 2:
            # # return
            
        # if self.ui.scan_tab.currentIndex() == 2:
            # self.img_sum_hist.clear()
            
            # # print('bins', self.doocs_bins)
            # # print('events', self.doocs_event_counts)
            
            # if len(self.doocs_bins) > 1:
                # self.img_sum_hist.plot(self.doocs_bins, self.doocs_event_counts, stepMode=True,  fillLevel=0,  brush=(100,100,100,150), clear=True)
                # self.img_sum_hist.setLabel('bottom', self.doocs_address_label, units=' ')
                # self.img_sum_hist.setTitle('{} events'.format(len(self.doocs_vals_hist_lagged)))
    
    def arange_spectra(self): #populate and trim spectrum array for analysis
        if self.ui.pb_start.text() == "Start" or not self.ui.analysis_acquire.isChecked():
            return
        print('arranging')
        
        transm = self.parent.get_transmission()
        
        n_shots_analysis = int(self.ui.analyze_last.value())
        print('before append: , self.spar.spec.shape=',self.spar.spec.shape)
        if len(self.spar.spec) == 1: #fresh unpopulated array
            print(' fresh unpopulated array')
            self.spar.spec = self.parent.spectrum_event[:, np.newaxis] * self.parent.calib_energy_coef / transm
            self.spar.phen = self.parent.x_axis
        elif self.spar.spec.shape[0] != len(self.parent.spectrum_event):
            print(' shapes inconsistent, refreshing')
            self.reset_spectra()
            return
        else:
            print(' all ok, old self.spar.spec.shape=', self.spar.spec.shape)
            self.spar.spec = np.append(self.spar.spec, self.parent.spectrum_event[:,np.newaxis] * self.parent.calib_energy_coef / transm, axis=1)
            self.spar.phen = self.parent.x_axis
            print('  new shape self.spar.spec.shape=',self.spar.spec.shape)
        # self.spec_hist.append(self.parent.spectrum_event)
        if n_shots_analysis > 0:
            if self.spar.events > n_shots_analysis:
                print('before cut: , self.spar.spec.shape=',self.spar.spec.shape)
                self.spar.spec = self.spar.spec[:,-n_shots_analysis:]
                print('after cut: , self.spar.spec.shape=',self.spar.spec.shape)
        print('shape(self.spar.spec)=',self.spar.spec.shape)
        print('shape(self.spar.phen)=',self.spar.phen.shape)
        
    def reset_spectra(self): #clears out spectra array
        self.spar = SpectrumArray()
        
    def add_spec_widget(self):
        print('adding spec_widget')
        win = pg.GraphicsLayoutWidget()
        layout = QtGui.QGridLayout()
        self.ui.widget_spectrum.setLayout(layout)
        layout.addWidget(win)

        self.img_spectrum = win.addPlot()
        self.img_spectrum.setLabel('left', 'intensity', units='arb.units')
        self.img_spectrum.setLabel('bottom', 'E_ph', units='eV')
        
        # self.reset_spec_image_widget()
    
    def plot_spec(self):
        if self.ui.pb_start.text() == "Stop" and self.ui.scan_tab.currentIndex() == 4:
            self.img_spectrum.clear()
            print('plotting, self.spar.spec.shape=',self.spar.spec.shape)
            if self.ui.pb_start.text() == "Start" or not self.ui.analysis_acquire.isChecked() or self.spar.events==0:
                return
            
            if self.spar.events == 1:
                speclast = specmin = specmax = specmean = self.spar.phen
            else:
                specmean = np.mean(self.spar.spec, axis=1)
                specmin = np.amin(self.spar.spec, axis=1)
                specmax = np.amax(self.spar.spec, axis=1)
                speclast = self.spar.spec[:,-1]
            print('specmean.shape=',specmean.shape)
            print('self.spar.phen.shape=',self.spar.phen.shape)
            
            pen_avg=pg.mkPen(color=(200, 0, 0), width=3)
            pen_single=pg.mkPen(color=(100, 100, 100), width=2)
            pen_lims=pg.mkPen(color=(200, 200, 200), width=1)
            self.img_spectrum.clear()
            self.img_spectrum.plot(self.spar.phen, specmean, stepMode=False, pen=pen_avg)
            self.img_spectrum.plot(self.spar.phen, speclast, stepMode=False, pen=pen_single)
            curvemax = self.img_spectrum.plot(self.spar.phen, specmax, stepMode=False, pen=pen_lims)
            curvemin = self.img_spectrum.plot(self.spar.phen, specmin, stepMode=False, pen=pen_lims)
            # fill = pg.
            # self.img_spectrum.FillBetweenItem(curvemin,curvemax,brush=(200,200,200,200))
            # self.img_spectrum.addItem(fill)
            
            E_ph_box = self.ui.analysis_Eph_box.value()
            dE_ph_box = self.ui.analysis_dEph_box.value()
            
            maxmean = specmax.max()
            
            # self.img_spectrum.plot([E_ph_box-dE_ph_box,E_ph_box+dE_ph_box], [maxmean, maxmean],fillLevel=0, brush=(50,100,100,200))
            
            # self.img_spectrum.plot([E_ph_box+dE_ph_box,E_ph_box+dE_ph_box], [0, maxmean], stepMode=False, pen=pen_lims)
            # self.img_dur = pg.ViewBox()
            # self.img_spectrum.scene().addItem(self.img_dur)
            # self.img_spectrum.getAxis('right').linkToView(self.img_dur)
            # self.img_dur.setXLink(self.img_spectrum)
            # self.img_spectrum.getAxis('right').setLabel('axis2')
            # # self.img_dur.setYRange(-10,10)
            # curve2 = pg.PlotCurveItem(pen=pg.mkPen(color='#025b94', width=1))
            # curve2.setData(x=self.spar.phen, y=np.ones_like(self.spar.phen))
            # self.img_dur.addItem(curve2)
            
            # self.img_dur.plot(self.spar.phen, np.ones_like(self.spar.phen), stepMode=False)
                
    
    def correlate(self):
        dE = self.ui.analysis_dEph_corr_box.value()
        self.corrn = self.spar.correlate_center(dE=dE)
        self.corrn.bin_phen(dE=dE)
        
        self.g2fit = self.corrn.fit_g2func(g2_gauss, thresh=0.1)

        corr_symm, domega = self.corrn.mirror()
        dphen = domega * hr_eV_s
        phen = self.corrn.omega * hr_eV_s
        # print(phen[0:5])
        # print(dphen[0:5])
        
    
    def add_hist_full_widget(self):
        print('adding hist_full widget')
        win = pg.GraphicsLayoutWidget()
        layout = QtGui.QGridLayout()
        self.ui.widget_histogram_full.setLayout(layout)
        layout.addWidget(win)

        self.histogram_full = win.addPlot()
        self.histogram_full.setLabel('bottom', 'intensity', units='arb.units')
        self.histogram_full.setLabel('left', 'events', units='_')
        
    def plot_hist_full(self):
        if self.ui.pb_start.text() == "Stop" and self.ui.scan_tab.currentIndex() == 4:
            # print('plotting, self.spar.spec.shape=',self.spar.spec.shape)
            if self.ui.pb_start.text() == "Start" or not self.ui.analysis_acquire.isChecked() or self.spar.events<2:
                return
            self.histogram_full.clear()
        
            nbins = int(self.ui.analysis_nbins_box.value())
            #TODO!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            
            W, W_hist, W_bins = self.spar.calc_histogram(bins=nbins, normed=0)

            Wm = numpy.mean(W) #average power calculated
            sigm2 = numpy.mean((W - Wm)**2) / Wm**2 #sigma square (power fluctuations)
            M_calc = 1 / sigm2 #calculated number of modes  
            
            
            # if self.spar.spec.shape[1] == 1:
                # speclast = specmin = specmax = specmean = self.spar.phen
            # else:
                # # specmean = np.mean(self.spar.spec, axis=1)
                # # specmin = np.amin(self.spar.spec, axis=1)
                # # specmax = np.amax(self.spar.spec, axis=1)
                # # speclast = self.spar.spec[:,-1]
            # print('specmean.shape=',specmean.shape)
            # print('self.phen.shape=',self.spar.phen.shape)
            
            # pen_avg=pg.mkPen(color=(200, 0, 0), width=3)
            # pen_single=pg.mkPen(color=(200, 200, 200), width=1)
            self.histogram_full.clear()
            self.histogram_full.plot(W_bins, W_hist, stepMode=True,  fillLevel=0,  brush=(100,100,100,150), clear=True)
            # self.histogram_full.plot(self.spar.phen, speclast, stepMode=False, pen=pen_single)


    def add_hist_peak_widget(self):
        print('adding hist_peak widget')
        win = pg.GraphicsLayoutWidget()
        layout = QtGui.QGridLayout()
        self.ui.widget_histogram_peak.setLayout(layout)
        layout.addWidget(win)

        self.histogram_peak = win.addPlot()
        self.histogram_peak.setLabel('bottom', 'intensity', units='arb.units')
        self.histogram_peak.setLabel('left', 'events', units='_')
        
    def plot_hist_peak(self):
        if self.ui.pb_start.text() == "Stop" and self.ui.scan_tab.currentIndex() == 4:
            # print('plotting, self.spar.spec.shape=',self.spar.spec.shape)
            if self.ui.pb_start.text() == "Start" or not self.ui.analysis_acquire.isChecked() or self.spar.events<2:
                return
            
            E_ph_box = self.ui.analysis_Eph_box.value()
            dE_ph_box = self.ui.analysis_dEph_box.value()
            nbins = int(self.ui.analysis_nbins_box.value())
            
            if E_ph_box == 0:
                if self.spar.events < 5:
                    return
                else:
                    specmean = np.mean(self.spar.spec, axis=1)
                    E_ph_val = self.spar.phen[specmean.argmax()]
            else:
                E_ph_val = E_ph_box
            
            self.histogram_peak.clear()
            
            try:
                W, W_hist, W_bins = self.spar.calc_histogram(E=[E_ph_val-dE_ph_box, E_ph_val+dE_ph_box], bins=nbins, normed=0)
            except ValueError:
                W_bins = np.arange(11)
                W_hist = np.zeros(10)
                W = np.zeros(10)
            
            Wm = numpy.mean(W) #average power calculated
            sigm2 = numpy.mean((W - Wm)**2) / Wm**2 #sigma square (power fluctuations)
            M_calc = 1 / sigm2 #calculated number of modes  
            # if self.spar.spec.shape[1] == 1:
                # speclast = specmin = specmax = specmean = self.spar.phen
            # else:
                # # specmean = np.mean(self.spar.spec, axis=1)
                # # specmin = np.amin(self.spar.spec, axis=1)
                # # specmax = np.amax(self.spar.spec, axis=1)
                # # speclast = self.spar.spec[:,-1]
            # print('specmean.shape=',specmean.shape)
            # print('self.phen.shape=',self.spar.phen.shape)
            
            # pen_avg=pg.mkPen(color=(200, 0, 0), width=3)
            # pen_single=pg.mkPen(color=(200, 200, 200), width=1)
            self.histogram_peak.clear()
            self.histogram_peak.plot(W_bins, W_hist, stepMode=True,  fillLevel=0,  brush=(100,100,100,150), clear=True)

    def add_durr_widget(self):
        win = pg.GraphicsLayoutWidget()
        layout = QtGui.QGridLayout()
        self.ui.widget_fit_pulse_dur.setLayout(layout)
        layout.addWidget(win)
        
        self.fit_pulse_dur = win.addPlot()
        self.fit_pulse_dur.setLabel('bottom', 'E_ph', units='eV')
        self.fit_pulse_dur.setLabel('left', 'duration', units='fs')
        
        # # self.add_corel2D_plot() #TMP~~~~~~~~~~~~~~~~~
        
        # # self.add_hist_plot()
        # #self.plot1.scene().sigMouseMoved.connect(self.mouseMoved)
        # # self.ui.cb_corel_spect.addItem("Peak Pos")
        # # self.ui.cb_corel_spect.addItem("Peak Ampl")
        # self.doocs_dev = None
        # # self.doocs_dev_hist = None
        # self.get_device()
        # self.ui.le_doocs_ch_cor2d.editingFinished.connect(self.get_device)
        
        
        # self.plot_timer = pg.QtCore.QTimer()
        # self.plot_timer.timeout.connect(self.plot_correl)
        # self.plot_timer.start(100)
        
        # self.plot_timer_hist_event = pg.QtCore.QTimer()
        # self.plot_timer_hist_event.timeout.connect(self.plot_hist_event)
        # self.plot_timer_hist_event.start(100)
        
        # self.plot_timer_Ipk_event = pg.QtCore.QTimer()
        # self.plot_timer_Ipk_event.timeout.connect(self.plot_Ipk_event)
        # self.plot_timer_Ipk_event.start(100)
        
        # self.plot_timer_Isum_event = pg.QtCore.QTimer()
        # self.plot_timer_Isum_event.timeout.connect(self.plot_Isum_event)
        # self.plot_timer_Isum_event.start(100)

        # # self.plot_timer_hist = pg.QtCore.QTimer()
        # # self.plot_timer_hist.timeout.connect(self.plot_histogram)
        # # self.plot_timer_hist.start(100)

        # self.phen = []
        # self.spec_hist = []
        # self.doocs_vals_hist = []
        
        # self.doocs_old_label = ''
        
        # self.spec_binned = []
        # self.doocs_bins = []
        # self.doocs_event_counts = []
        # self.doocs_vals_hist_lagged = []
        
        # self.event_counter = 0
        
        # self.doocs_address_label = self.ui.le_doocs_ch_cor2d.text()
        
        # # self.channel_timer = pg.QtCore.QTimer()
        # # if self.doocs_address_label != self.ui.le_doocs_ch_cor2d.text():
            # # self.reset()
            # # self.doocs_address_label = self.ui.le_doocs_ch_cor2d.text()
        # # self.channel_timer.start(100)
        
        
        # self.ui.sb_corr_2d_reset.clicked.connect(self.reset)

        # # self.ui.pb_start_scan.clicked.connect(self.start_stop_scan)
        # # self.ui.pb_check_range.clicked.connect(self.check_range)
        # # self.ui.pb_show_map.clicked.connect(self.show_hide_map)
        
        # self.doocs_address_label = '' #костыль
        # self.add_corr2d_image_widget()
        # self.add_hist_event_widget()
        # self.add_corrIpk_widget()
        # self.add_corrIsum_widget()
        
        # self.ui.actionSave_Corelation.triggered.connect(self.save_corr2d_data_as)

    # def stop_timers(self):
        # self.plot_timer.stop()
        # self.plot_timer_hist_event.stop()
        # self.plot_timer_Ipk_event.stop()
        # self.plot_timer_Isum_event.stop()

    # def get_device(self):
        # if self.ui.is_le_addr_ok(self.ui.le_doocs_ch_cor2d):
            # eid = self.ui.le_doocs_ch_cor2d.text()
            # self.doocs_dev = Device(eid=eid)
            # self.doocs_dev.mi = self.mi
        # else:
            # self.doocs_dev = None
            
    
    # def sort_and_bin(self):
        
        
        # try:
            # bin_doocs = float(self.ui.sb_corr2d_binning.text()) #bin size
        # except ValueError:
            # bin_doocs = 0
            
        # try:
            # phen_min = self.ui.sb_emin.value()
        # except ValueError:
            # phen_min = -np.inf
            
        # self.phen_orig = self.parent.x_axis
            
        # try:
            # phen_max = self.ui.sb_emax.value()
        # except ValueError:
            # phen_max = np.inf
            
        # # print('phen_max',phen_max)
        # # print('phen_min',phen_min)
        
        # d2, d1 = 0, 0
        
        # if phen_max > phen_min:
            # d1 = find_nearest_idx(self.phen_orig/1000, phen_min)
            # d2 = find_nearest_idx(self.phen_orig/1000, phen_max)
        # # else:
        # if d2 <= d1:
            # d1 = 0
            # d2 = len(self.phen_orig)
        
        
        
        # print('d1, d2', d1, d2)
        # self.phen = self.phen_orig[d1:d2]
        # n_phens = len(self.phen)
        
        # print('self.phen_orig',len(self.phen_orig))
        # print('self.phen',len(self.phen))
        
        # if bin_doocs==0:
            # bin_doocs=1e10
            
        # try:
            # self.n_lag = int(self.ui.sb_corr2d_lag.value()) #lag size
        # except ValueError:
            # self.n_lag = 0
        
        # print('len(self.doocs_vals_hist)', len(self.doocs_vals_hist))
        # if len(self.doocs_vals_hist) > abs(self.n_lag)+5:
            # if self.n_lag >= 0:
                # self.doocs_vals_hist_lagged = self.doocs_vals_hist[:len(self.doocs_vals_hist)-self.n_lag]
                # spec_lagged = np.array(self.spec_hist)[self.n_lag:, :]
            # else:
                # self.doocs_vals_hist_lagged = self.doocs_vals_hist[abs(self.n_lag):]
                # spec_lagged = np.array(self.spec_hist)[:len(self.doocs_vals_hist)-abs(self.n_lag), :]
                
        # else:
            # self.doocs_vals_hist_lagged = self.doocs_vals_hist
            # spec_lagged = np.array(self.spec_hist)
        
        
        
        # # spec = np.array(self.spec_hist)
        # # min_val = bin_doocs * (int(min(self.doocs_vals_hist) / bin_doocs))
        # # max_val = max(self.doocs_vals_hist)
        # min_val = bin_doocs * (int(min(self.doocs_vals_hist_lagged) / bin_doocs)) #ensures the minimum value is integer of bin width and figure does not jitter
        # max_val = max(self.doocs_vals_hist_lagged) + bin_doocs * 1.01
        
        # if max_val - min_val <= bin_doocs:
            # max_val = min_val + 1.01 * bin_doocs #ensures there is at least one bin (two bin values)
        
        # # if min_val == max_val:
            # # max_val = 1.001 * min_val #ensures there is at least one bin
        
        # print('min_DOOCS_val', min_val)
        # print('max_DOOCS_val', max_val)
        # # print('bin_doocs', bin_doocs)
        # self.doocs_bins = np.arange(min_val, max_val, bin_doocs)
        # # print('shape of created doocs_bins', self.doocs_bins.shape)
        
        # self.doocs_event_counts, _ = np.histogram(self.doocs_vals_hist_lagged, bins=self.doocs_bins)
        
        # print('len spec_hist',len(self.spec_hist))
        # print('len doocs_vals_hist',len(self.doocs_vals_hist))
        
        # print('shape of spec_lagged array', spec_lagged.shape)
        # print('shape of doocs_bins', self.doocs_bins.shape)
        # print('doocs_bins',len(self.doocs_bins), self.doocs_bins)
        
        # self.bin_dest_idx = np.digitize(self.doocs_vals_hist_lagged, self.doocs_bins)-1
        # self.spec_binned = np.zeros((len(self.doocs_bins)-1, n_phens))
        
        # print('self.bin_dest_idx', self.bin_dest_idx)
        # print('self.spec_binned', len(self.spec_binned), self.spec_binned)
        # print('spec_lagged', len(spec_lagged), spec_lagged)
        # print('self.doocs_vals_hist_lagged', len(self.doocs_vals_hist_lagged), self.doocs_vals_hist_lagged)
        
        # for i in np.unique(self.bin_dest_idx):
            # idx = np.where(i == self.bin_dest_idx)[0]
            # # print('sorting', i, idx)
            # if len(idx) > 1:
                # self.spec_binned[i, :] = np.mean(spec_lagged[idx, d1:d2], axis=0)
                # print('multiple', i, idx, len(self.spec_binned))
            # elif len(idx) == 1:
                # print('singe', i, idx, len(self.spec_binned))
                # # if len(self.spec_binned) != 0:
                # print('self.spec_binned[i, :]', self.spec_binned[i, :])
                # print('spec_lagged[idx[0], d1:d2]', spec_lagged[idx[0], d1:d2])
                # self.spec_binned[i, :] = spec_lagged[idx[0], d1:d2]
                # # else:
                    # # self.spec_binned = np.array([spec_lagged[idx[0], :]])
            # else:
                # pass
        
        # print('self.spec_binned', self.spec_binned)
        # print('self.doocs_bins', self.doocs_bins)
        # # if 
        
        # # self.doocs_event_counts = np.unique(self.bin_dest_idx, return_counts=1)[1]
        
        
    # def reset(self):
        # self.spec_hist = []
        # self.doocs_vals_hist = []
        # self.img_hist.clear()
        # self.img_corr2d.clear()
        # self.doocs_address_label = self.ui.le_doocs_ch_cor2d.text()
        
    # def plot_correl(self):
        
        # self.event_counter += 1
        
        # if self.ui.pb_start.text() == "Start" or not self.ui.sb_corr_2d_run.isChecked() or self.parent.spectrum_event is None:
            # return
            
        # # print('DOOCS LABEL ',self.doocs_address_label)
        # if self.ui.le_doocs_ch_cor2d.text() != self.doocs_address_label:
            # self.reset()
            # return
        
        
        # # print('min_self.phen_val', min(self.phen))
        # # print('max_self.phen_val', max(self.phen))
        
        # n_shots = int(self.ui.sb_n_shots_max.value())
        # if len(self.spec_hist) > n_shots: #add lag value
            # self.spec_hist = self.spec_hist[-n_shots:]
            # self.doocs_vals_hist = self.doocs_vals_hist[-n_shots:]
        
        # self.spec_hist.append(self.parent.spectrum_event)
        
        
        # #print(self.doocs_address_label)
        
        # if self.doocs_address_label == 'event':
            # self.doocs_vals_hist.append(self.event_counter)
        # elif self.doocs_address_label == 'dummy label':
            # self.doocs_vals_hist.append(np.sin(time.time()/10)*7.565432 + 25)
        # elif self.parent.ui.combo_hirex.currentText() != "DUMMY":
            # if self.doocs_dev is None:
                # self.ui.sb_corr_2d_run.setChecked(False)
                # self.parent.error_box("Wrong DOOCS channel")
                # return
            # self.doocs_vals_hist.append(self.doocs_dev.get_value())
        # else:
            # self.doocs_address_label = 'event',
            # self.doocs_vals_hist.append(self.event_counter)

        # # print(self.doocs_vals_hist[-1])
        
        # if self.ui.scan_tab.currentIndex() == 2:
            # self.sort_and_bin()
            # scale_yaxis = (self.phen[-1] - self.phen[0]) / len(self.phen)
            # translate_yaxis = self.phen[0] / scale_yaxis
            
            # scale_xaxis = (max(self.doocs_bins) - min(self.doocs_bins)) / len(self.doocs_bins)
            # translate_xaxis = min(self.doocs_bins) / scale_xaxis
            # # self.single_scatter.setData(self.peak, self.doocs_vals)
            
            # self.add_corr2d_image_item()
            # self.img.setImage(self.spec_binned)
            # # print('spec binned',len(self.spec_binned))
            # # print('doocs_bins',len(self.doocs_bins))
            # #elegant but maybe wrong
            # # rect = QtCore.QRect(min(self.doocs_bins), min(self.phen), max(self.doocs_bins)-min(self.doocs_bins), max(self.phen)-min(self.phen))
            # # self.img.setRect(rect)
            # self.img.scale(scale_xaxis, scale_yaxis)
            # self.img.translate(translate_xaxis, translate_yaxis)
            # self.img_corr2d.setLabel('bottom', self.doocs_address_label, units='_')
            
        # self.doocs_old_label = self.doocs_address_label
        

    # def add_corr2d_image_widget(self):
        # win = pg.GraphicsLayoutWidget()
        # layout = QtGui.QGridLayout()
        # self.ui.widget_corr2d.setLayout(layout)
        # layout.addWidget(win)

        # self.img_corr2d = win.addPlot()
        # self.add_corr2d_image_item()
        
    # def add_hist_event_widget(self):
        # gui_index = self.ui.get_style_name_index()
        # if "standard" in self.parent.gui_styles[gui_index]:
            # pg.setConfigOption('background', 'w')
            # pg.setConfigOption('foreground', 'k')
            # single_pen = pg.mkPen("k")
        # else:
            # single_pen = pg.mkPen("w")
        # win = pg.GraphicsLayoutWidget()
        
        # layout = QtGui.QGridLayout()
        # self.ui.widget_corr_event.setLayout(layout)
        # layout.addWidget(win)

        # self.img_hist = win.addPlot()
        # self.img_hist.setLabel('left', "N Events", units='')
        # # self.img_hist.showGrid(1, 1, 1)
        # self.img_hist.setLabel('bottom', self.doocs_address_label, units='_')
        
    # def add_corr2d_image_item(self):
        # self.img_corr2d.clear()

        # self.img_corr2d.setLabel('left', "E_ph", units='eV')
        # self.img_corr2d.setLabel('bottom', self.doocs_address_label, units='_')

        # self.img = pg.ImageItem()
        # self.img_corr2d.addItem(self.img)

        # colormap = cm.get_cmap('viridis') #"nipy_spectral")  # cm.get_cmap("CMRmap")
        # colormap._init()
        # lut = (colormap._lut * 255).view(np.ndarray)  # Convert matplotlib colormap from 0-1 to 0 -255 for Qt
        # # Apply the colormap
        # self.img.setLookupTable(lut)
        
    # def plot_hist_event(self):
    
        # if self.ui.pb_start.text() == "Start" or not self.ui.sb_corr_2d_run.isChecked():
            # return
        
        # # if len(self.doocs_vals_hist_lagged) < 2:
            # # return
            
        # if self.ui.scan_tab.currentIndex() == 2:
            # self.img_hist.clear()
            
            # # print('bins', self.doocs_bins)
            # # print('events', self.doocs_event_counts)
            
            # if len(self.doocs_bins) > 1:
                # self.img_hist.plot(self.doocs_bins, self.doocs_event_counts, stepMode=True,  fillLevel=0,  brush=(100,100,100,150), clear=True)
                # self.img_hist.setLabel('bottom', self.doocs_address_label, units=' ')
                # self.img_hist.setTitle('{} events'.format(len(self.doocs_vals_hist_lagged)))
        # # self.img = pg.ImageItem()
        # # self.img_hist.addItem(self.img)

        # # colormap = cm.get_cmap('viridis') #"nipy_spectral")  # cm.get_cmap("CMRmap")
        # # colormap._init()
        # # lut = (colormap._lut * 255).view(np.ndarray)  # Convert matplotlib colormap from 0-1 to 0 -255 for Qt
        # # Apply the colormap
    
            
    # def add_corrIpk_widget(self):
        # gui_index = self.ui.get_style_name_index()
        # if "standard" in self.parent.gui_styles[gui_index]:
            # pg.setConfigOption('background', 'w')
            # pg.setConfigOption('foreground', 'k')
            # single_pen = pg.mkPen("k")
        # else:
            # single_pen = pg.mkPen("w")
        # win = pg.GraphicsLayoutWidget()
        
        # layout = QtGui.QGridLayout()
        # self.ui.widget_corr_Ipeak.setLayout(layout)
        # layout.addWidget(win)

        # self.img_Ipk = win.addPlot()
        # self.img_Ipk.setLabel('left', "I (peak)", units='')
        # # self.img_Ipk.showGrid(1, 1, 1)
        # self.img_Ipk.setLabel('bottom', self.doocs_address_label, units='_')
        
    # def plot_Ipk_event(self):
        # if self.ui.pb_start.text() == "Start" or not self.ui.sb_corr_2d_run.isChecked():
            # return
        # if self.ui.scan_tab.currentIndex() == 2:
            # self.img_Ipk.clear()
            # if len(self.doocs_bins) > 1:
                
                # bin_step = self.doocs_bins[1] - self.doocs_bins[0]
                # interbin_values = self.doocs_bins[:-1] + bin_step/2
                
                # # print('plot_Ipk', len(interbin_values), len(self.spec_binned))
                # pen=pg.mkPen(color=(255, 0, 0), width=3)
                # self.img_Ipk.plot(interbin_values, np.amax(self.spec_binned,axis=1), stepMode=False, pen=pen)#,  fillLevel=0,  brush=(0,0,255,150), clear=True)
                # self.img_Ipk.setLabel('bottom', self.doocs_address_label, units=' ')
                # # self.img_Ipk.setLimits(yMin=0)
        
    # def add_corrIsum_widget(self):
        # gui_index = self.ui.get_style_name_index()
        # if "standard" in self.parent.gui_styles[gui_index]:
            # pg.setConfigOption('background', 'w')
            # pg.setConfigOption('foreground', 'k')
            # single_pen = pg.mkPen("k")
        # else:
            # single_pen = pg.mkPen("w")
        # win = pg.GraphicsLayoutWidget()
        
        # layout = QtGui.QGridLayout()
        # self.ui.widget_corr_Isum.setLayout(layout)
        # layout.addWidget(win)

        # self.img_Isum = win.addPlot()
        # self.img_Isum.setLabel('left', "Ipk/Isum", units='')
        # # self.img_Isum.showGrid(1, 1, 1)
        # self.img_Isum.setLabel('bottom', self.doocs_address_label, units=' ')
        
    # def plot_Isum_event(self):
        # if self.ui.pb_start.text() == "Start" or not self.ui.sb_corr_2d_run.isChecked():
            # return
        # if self.ui.scan_tab.currentIndex() == 2:
            # self.img_Isum.clear()
            # if len(self.doocs_bins) > 1:
                
                # bin_step = self.doocs_bins[1] - self.doocs_bins[0]
                # interbin_values = self.doocs_bins[:-1] + bin_step/2
                
                # # print('plot_Ipk', len(interbin_values), len(self.spec_binned))
                # pen=pg.mkPen(color=(0, 0, 255), width=3)
                # self.img_Isum.plot(interbin_values, np.amax(self.spec_binned,axis=1)/np.sum(self.spec_binned,axis=1), stepMode=False, pen=pen)#,  fillLevel=0,  brush=(0,0,255,150), clear=True)
                # self.img_Isum.setLabel('bottom', self.doocs_address_label, units=' ')
        
    # def save_corr2d_data_as(self):
        # filename = QtGui.QFileDialog.getSaveFileName(self.parent, 'Save Correlation&Spectrum Data',
                                                     # self.parent.data_dir, "txt (*.npz)", None,
                                                     # QtGui.QFileDialog.DontUseNativeDialog
                                                     # )[0]
        # np.savez(filename, phen_scale=self.phen, spec_hist=self.spec_hist, doocs_vals_hist=self.doocs_vals_hist, corr2d=self.spec_binned, doocs_scale = self.doocs_bins, doocs_channel=self.doocs_address_label)