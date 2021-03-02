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
        
        self.spec_axis_label = 'arb.units' #temp, improve
        
        self.spar = SpectrumArray()
        self.g2fit = FitResult()
        self.g2fit.omega=np.array([0])
        
        self.n_events_processed = 0
        self.n_last_correlated = 0
        
        self.acquire_timer = pg.QtCore.QTimer()
        self.acquire_timer.timeout.connect(self.arange_spectra)
        self.acquire_timer.start(100)
        
        
        self.plot_timer = pg.QtCore.QTimer()
        self.plot_timer.timeout.connect(self.plot_spec)
        self.plot_timer.timeout.connect(self.plot_hist_full)
        self.plot_timer.timeout.connect(self.plot_hist_peak)
        self.plot_timer.timeout.connect(self.print_n_events)
        self.plot_timer.timeout.connect(self.correlate_and_plot_auto)
        self.plot_timer.start(1000)
        
        self.add_spec_widget()
        self.add_hist_full_widget()
        self.add_hist_peak_widget()
        self.add_durr_widget()
        
        self.ui.analysis_resetbutton.clicked.connect(self.reset_spectra)
        
        self.ui.analysis_correlate_button.clicked.connect(self.correlate_and_plot)
    
    def worth_plotting(self):
        started = self.ui.pb_start.text() == "Stop"
        current_tab = self.ui.scan_tab.currentIndex() == 4
        analysis_acquiring = self.ui.analysis_acquire.isChecked()
        if started and current_tab and analysis_acquiring:
            return True
        else:
            return False
        
    
    
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
        # print('arranging')
        
        if self.ui.chb_uj_ev.isChecked():
            self.parent.get_transmission()
            energy_calib = self.parent.calib_energy_coef * 1e-6 #convert to Joules 
            transm = self.parent.transmission_value
            self.spec_axis_label = 'J/eV'
        else:
            energy_calib = transm = 1
            self.spec_axis_label = 'arb.units'
        
        n_shots_analysis = int(self.ui.analyze_last.value())
        # print('before append: , self.spar.spec.shape=',self.spar.spec.shape)
        if len(self.spar.spec) == 1: #fresh unpopulated array
            # print(' fresh unpopulated array')
            self.spar.spec = self.parent.spectrum_event[:, np.newaxis] * self.parent.calib_energy_coef / transm
            self.spar.phen = self.parent.x_axis
        elif self.spar.spec.shape[0] != len(self.parent.spectrum_event):
            # print(' shapes inconsistent, refreshing')
            self.reset_spectra()
            return
        else:
            # print(' all ok, old self.spar.spec.shape=', self.spar.spec.shape)
            self.spar.spec = np.append(self.spar.spec, self.parent.spectrum_event[:,np.newaxis] * self.parent.calib_energy_coef / transm, axis=1)
            self.spar.phen = self.parent.x_axis
            # print('  new shape self.spar.spec.shape=',self.spar.spec.shape)
        # self.spec_hist.append(self.parent.spectrum_event)
        if n_shots_analysis > 0:
            if self.spar.events > n_shots_analysis:
                # print('before cut: , self.spar.spec.shape=',self.spar.spec.shape)
                self.spar.spec = self.spar.spec[:,-n_shots_analysis:]
                # print('after cut: , self.spar.spec.shape=',self.spar.spec.shape)
        # print('shape(self.spar.spec)=',self.spar.spec.shape)
        # print('shape(self.spar.phen)=',self.spar.phen.shape)
        
        self.n_events_processed += 1
        
    def reset_spectra(self): #clears out spectra array
        self.spar = SpectrumArray()
        
    def add_spec_widget(self):
        # print('adding spec_widget')
        win = pg.GraphicsLayoutWidget()
        layout = QtGui.QGridLayout()
        self.ui.widget_spectrum.setLayout(layout)
        layout.addWidget(win)

        self.img_spectrum = win.addPlot()
        self.img_spectrum.setLabel('left', 'intensity', units=self.spec_axis_label)
        self.img_spectrum.setLabel('bottom', 'E_ph', units='eV')
        
        pen_avg=pg.mkPen(color=(200, 0, 0), width=3)
        pen_single=pg.mkPen(color=(100, 100, 100), width=2)
        pen_lims=pg.mkPen(color=(200, 200, 200), width=1)
        
        self.spec_mean_curve = self.img_spectrum.plot(stepMode=False, pen=pen_avg)
        self.spec_last_curve = self.img_spectrum.plot(stepMode=False, pen=pen_single)
        self.spec_max_curve = self.img_spectrum.plot(stepMode=False, pen=pen_lims)
        self.spec_min_curve = self.img_spectrum.plot(stepMode=False, pen=pen_lims)
        
        # win1 = pg.plot()
        # fill = pg.FillBetweenItem(self.spec_min_curve, self.spec_max_curve, brush='k', pen='k')
        # win1.addItem(fill)
        # pg.FillBetweenItem(self.spec_min_curve, self.spec_max_curve, brush=(200,200,200,200))
    
    def plot_spec(self):
        if self.worth_plotting():
            
            if self.spar.events == 1:
                speclast = specmin = specmax = specmean = self.spar.spec[:,-1]
            else:
                specmean = np.mean(self.spar.spec, axis=1)
                specmin = np.amin(self.spar.spec, axis=1)
                specmax = np.amax(self.spar.spec, axis=1)
                speclast = self.spar.spec[:,-1]
            # print('specmean.shape=',specmean.shape)
            # print('self.spar.phen.shape=',self.spar.phen.shape)
            
            self.spec_mean_curve.setData(self.spar.phen, specmean)
            self.spec_last_curve.setData(self.spar.phen, speclast)
            self.spec_max_curve.setData(self.spar.phen, specmax)
            self.spec_min_curve.setData(self.spar.phen, specmin)
            
            
            
            # curvemin = self.img_spectrum.plot(self.spar.phen, specmin, stepMode=False, pen=pen_lims)
            # fill = pg.
            
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
            
            # print('self.n_last_correlated', self.n_last_correlated)
            # print('self.n_events_processed', self.n_events_processed)
            # print('self.ui.spinbox_correlate_every.value()', self.ui.spinbox_correlate_every.value())
    
    def correlate(self):
        dE = self.ui.analysis_dEph_corr_box.value()
        self.corrn = self.spar.correlate_center(dE=dE, norm=1)
        self.corrn.bin_phen(dE=dE)
        try: 
            self.g2fit = self.corrn.fit_g2func(g2_gauss, thresh=0.1)
        except:
            pass
        self.n_last_correlated = self.n_events_processed
        # corr_symm, domega = self.corrn.mirror()
        # dphen = domega * hr_eV_s
        # phen = self.corrn.omega * hr_eV_s
        # print(phen[0:5])
        # print(dphen[0:5])
        
    
    def add_hist_full_widget(self):
        # print('adding hist_full widget')
        win = pg.GraphicsLayoutWidget()
        layout = QtGui.QGridLayout()
        self.ui.widget_histogram_full.setLayout(layout)
        layout.addWidget(win)

        self.histogram_full = win.addPlot()
        self.histogram_full.setLabel('bottom', 'intensity', units=self.spec_axis_label)
        self.histogram_full.setLabel('left', 'full events', units='')
        
    def plot_hist_full(self):
        if self.worth_plotting():
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
        # print('adding hist_peak widget')
        win = pg.GraphicsLayoutWidget()
        layout = QtGui.QGridLayout()
        self.ui.widget_histogram_peak.setLayout(layout)
        layout.addWidget(win)

        self.histogram_peak = win.addPlot()
        self.histogram_peak.setLabel('bottom', 'intensity', units=self.spec_axis_label)
        self.histogram_peak.setLabel('left', 'window events', units='')
        
    def plot_hist_peak(self):
        if self.worth_plotting():
            
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
        self.fit_pulse_dur.setLabel('left', 'duration', units='s')
        self.durr_curve = self.fit_pulse_dur.plot(symbolBrush='r', symbolPen='r')
        self.fit_pulse_dur.setXLink(self.img_spectrum)
        # self.fit_pulse_dur.setYRange(0,2)
        
    def update_durr_plot(self):
        idx = self.g2fit.fit_t>0
        dur = self.g2fit.fit_t * self.g2fit.fit_pedestal / self.g2fit.fit_contrast
        self.durr_curve.setData(self.g2fit.omega[idx]* hr_eV_s, dur[idx])
        # self.fit_pulse_dur.setLimits(yMin=0)
        # self.fit_pulse_dur.setLimits([0,2])
        
        # print('pedestal')
        # print(self.g2fit.fit_pedestal)
        # print('contrast')
        # print(self.g2fit.fit_contrast)
        
    def correlate_and_plot(self):
        self.correlate()
        self.update_durr_plot()
        
    def correlate_and_plot_auto(self):
        n_new = self.n_events_processed - self.n_last_correlated
        correlate_every = self.ui.spinbox_correlate_every.value()
        if correlate_every > 0 and n_new > correlate_every:
            self.correlate_and_plot()
        
        
    def print_n_events(self):
        self.ui.label_30.setText("Process last {} of ".format(self.spar.events))
