import numpy as np
import copy
import pyqtgraph as pg
from threading import Thread, Event
from PyQt5 import QtGui, QtCore
import time
from mint.opt_objects import Device
from scipy import ndimage
from matplotlib import cm
from opt_lib import *

# from opt_lib import SpectrumArray

#fixing pyqtgraph poor coding practice
step_arg = True
#pyqtgraph_v = pg.__version__.split('.')
#if int(pyqtgraph_v[0]) == 0 and int(pyqtgraph_v[1]) < 11:
#    step_arg = True
#else:
#    step_arg = 'center'

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
        self.spar_screwed = SpectrumArray()
        self.g2fit = FitResult()
        self.g2fit.omega=np.array([0])
        self.g2fit.fit_t_comp=np.array([0])
        self.g2fit.fit_t=np.array([0])
        # self.g2_plot_idx = 0
        
        self.n_events_processed = 0
        self.n_last_correlated = 0
        
        
        self.acquire_timer = pg.QtCore.QTimer()
        self.acquire_timer.timeout.connect(self.arange_spectra)
        self.acquire_timer.timeout.connect(self.get_Eph_box_values)
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
        self.add_g2_line_widget()
        self.add_rosa_widget()
        
        self.reset_spectra()
        self.ui.analysis_resetbutton.clicked.connect(self.clear_all_curves)
        self.ui.analysis_resetbutton.clicked.connect(self.reset_spectra)
        
        self.phen_last = np.array([])
        
        self.ui.analysis_correlate_button.clicked.connect(self.correlate_and_plot)
    
    def get_Eph_box_values(self):
        self.E_ph_box = self.ui.analysis_Eph_box.value()
        self.dE_ph_box = self.ui.analysis_dEph_box.value()
        self.hist_nbins = int(self.ui.analysis_nbins_box.value())
        if self.hist_nbins == 0:
            self.hist_nbins = 1
        if self.E_ph_box == 0:
            if self.spar.events < 2:
                self.E_ph_box_used = 0
                return
            else:
                specmean = np.mean(self.spar.spec, axis=1)
                self.E_ph_box_used = self.spar.phen[specmean.argmax()]
        else:
            self.E_ph_box_used = self.E_ph_box
    
    def worth_plotting(self):
        started = self.ui.pb_start.text() == "Stop"
        current_tab = self.ui.scan_tab.currentIndex() == 4
        analysis_acquiring = self.ui.analysis_acquire.isChecked()
        if started and current_tab and analysis_acquiring and self.hist_nbins>2:
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
        if self.ui.pb_start.text() == "Start" or self.parent.spectrum_event_disp is None or not self.ui.analysis_acquire.isChecked():
            return
        # print('arranging')
        
        # wrong_size = self.spar.spec.shape[0] != len(self.parent.spectrum_event_disp)
        # shifted_spec = self.spar.phen[0] != self.parent.x_axis_disp[0] or self.spar.phen[-1] != self.parent.x_axis_disp[-1]
        
        
        
        if not np.array_equal(self.phen_last, self.parent.x_axis_disp):
            print('different axis, skipping')
            self.reset_spectra()
            self.phen_last = self.parent.x_axis_disp
            return
            
        
        #if shifted_spec:
        #    print('correlation analysis: photon energy scale changed')
        #    self.reset_spectra()
        zeroscale = self.parent.x_axis_disp[-1] == self.parent.x_axis_disp[0]
        
        if zeroscale:
            print('correlation analysis: x_axis[-1] == x_axis[0]')
            self.reset_spectra()
            return
        
        if self.ui.chb_uj_ev.isChecked():
            # if self.spec_axis_label == 'arb.units' or energy_calib == 1:
                # self.reset_spectra()
            energy_calib = self.parent.calib_energy_coef * 1e-6 #convert to Joules 
            self.parent.get_transmission()
            transm = self.parent.transmission_value
            self.spec_axis_label = 'J/eV'
        else:
            # if self.spec_axis_label == 'J/eV' or energy_calib == self.parent.calib_energy_coef * 1e-6:
                # self.reset_spectra()
            energy_calib = transm = 1
            self.spec_axis_label = 'arb.units'
        
        n_shots_analysis = int(self.ui.analyze_last.value())
        # print('before append: , self.spar.spec.shape=',self.spar.spec.shape)
        if len(self.spar.spec) == 1: #fresh unpopulated array
            # print(' fresh unpopulated array')
            self.spar.spec = self.parent.spectrum_event_disp[:, np.newaxis] * self.parent.calib_energy_coef / transm
            self.spar.phen = self.parent.x_axis_disp
        else:
            # print(' all ok, old self.spar.spec.shape=', self.spar.spec.shape)
            self.spar.spec = np.append(self.spar.spec, self.parent.spectrum_event_disp[:,np.newaxis] * self.parent.calib_energy_coef / transm, axis=1)
            self.spar.phen = self.parent.x_axis_disp
            # print('  new shape self.spar.spec.shape=',self.spar.spec.shape)
        # self.spec_hist.append(self.parent.spectrum_event_disp)
        if n_shots_analysis > 0:
            if self.spar.events > n_shots_analysis:
                # print('before cut: , self.spar.spec.shape=',self.spar.spec.shape)
                self.spar.spec = self.spar.spec[:,-n_shots_analysis:]
                # print('after cut: , self.spar.spec.shape=',self.spar.spec.shape)
        # print('shape(self.spar.spec)=',self.spar.spec.shape)
        # print('shape(self.spar.phen)=',self.spar.phen.shape)
        
        self.n_events_processed += 1
        
        self.spar_screwed = copy.deepcopy(self.spar)
        if self.ui.box_convolve.value() > 0:
            self.spar_screwed.conv_gauss(dE=self.ui.box_convolve.value())
        if self.ui.box_ephjitter.value() > 0:
            self.spar_screwed.add_jitter_ev(self.ui.box_ephjitter.value())
        if self.ui.box_epulsejitter.value() > 0:
            self.spar_screwed.add_jitter_en(self.ui.box_epulsejitter.value())
        if self.ui.box_pedestal.value() > 0:
            self.spar_screwed.spec += self.ui.box_pedestal.value()
        
    def reset_spectra(self): #clears out spectra array
        self.spar = SpectrumArray()
        self.spar_screwed = SpectrumArray()
        self.corrn = SpectrumCorrelationsCenter()
        self.g2fit = FitResult()
        self.g2fit.fit_t_comp=np.array([0])
        
    def add_spec_widget(self):
        # print('adding spec_widget')
        win = pg.GraphicsLayoutWidget()
        layout = QtGui.QGridLayout()
        self.ui.widget_spectrum.setLayout(layout)
        layout.addWidget(win)

        self.img_spectrum = win.addPlot()
        self.img_spectrum.addLegend()
        self.img_spectrum.setLabel('left', 'intensity', units=self.spec_axis_label)
        self.img_spectrum.setLabel('bottom', '<math>E<sub>ph</sub></math>', units='eV')
        
        pen_avg=pg.mkPen(color=(200, 0, 0), width=3)
        pen_single=pg.mkPen(color=(100, 100, 100), width=2)
        pen_lims=pg.mkPen(color=(200, 200, 200), width=1)
        pen_wlims=pg.mkPen(color=(0, 200, 0), width=2)
        
        self.spec_mean_curve = self.img_spectrum.plot(stepMode=False, pen=pen_avg, name='mean')
        self.spec_last_curve = self.img_spectrum.plot(stepMode=False, pen=pen_single, name='singleshot')
        self.spec_max_curve = self.img_spectrum.plot(stepMode=False, pen=pen_lims, name='limits')
        self.spec_min_curve = self.img_spectrum.plot(stepMode=False, pen=pen_lims)
        
        self.spec_window_r = self.img_spectrum.plot(pen=pen_wlims)
        self.spec_window_l = self.img_spectrum.plot(pen=pen_wlims, name='hist window')
        
        
        
        # win1 = pg.plot()
        # fill = pg.FillBetweenItem(self.spec_min_curve, self.spec_max_curve, brush='k', pen='k')
        # win1.addItem(fill)
        # pg.FillBetweenItem(self.spec_min_curve, self.spec_max_curve, brush=(200,200,200,200))
    
    def plot_spec(self):
        if self.worth_plotting():
            spar = self.spar_screwed #############################TMP###############
            # spar = self.spar
            
            if spar.events == 1:
                speclast = specmin = specmax = specmean = spar.spec[:,-1]
            else:
                specmean = np.mean(spar.spec, axis=1)
                specmin = np.amin(spar.spec, axis=1)
                specmax = np.amax(spar.spec, axis=1)
                speclast = spar.spec[:,-1]
                
            max_spec = np.amax(specmax)
            # print('specmean.shape=',specmean.shape)
            # print('spar.phen.shape=',spar.phen.shape)
            
            self.spec_mean_curve.setData(spar.phen, specmean)
            self.spec_last_curve.setData(spar.phen, speclast)
            self.spec_max_curve.setData(spar.phen, specmax)
            self.spec_min_curve.setData(spar.phen, specmin)
            
            # E_ph_box = self.ui.analysis_Eph_box.value()
            # dE_ph_box = self.ui.analysis_dEph_box.value()
            
            self.spec_window_l.setData([self.E_ph_box_used-self.dE_ph_box,self.E_ph_box_used-self.dE_ph_box],[0,max_spec])
            self.spec_window_r.setData([self.E_ph_box_used+self.dE_ph_box,self.E_ph_box_used+self.dE_ph_box],[0,max_spec])
            
            # curvemin = self.img_spectrum.plot(spar.phen, specmin, stepMode=False, pen=pen_lims)
            # fill = pg.
            
            # self.img_spectrum.addItem(fill)
            
            
            
            
            
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
        spar = self.spar_screwed #############################TMP###############
        if dE == 0:
            print('dE == 0, not correlating')
            self.n_last_correlated = 0
            return
        
        if spar.events < 2 or len(spar.phen)<4:
            print('spar.events < 2 or len(spar.phen)<4, not correlating')
            self.n_last_correlated = 0
            return
        
        # try: 
        self.corrn = spar.correlate_center(dE=dE, norm=1)
        self.corr = spar.correlate_center(dE=dE, norm=0)
        self.corrn.bin_phen(dE=dE)
        self.corr.bin_phen(dE=dE)
        
        self.reconstr = self.corr.fft()
        
        if len(self.corrn.dphen) < 4:
            print('too little points for fit')
            self.n_last_correlated = 0
            return
        self.g2fit = self.corrn.fit_g2func(g2_gauss, thresh=0.1)
        # print("self.g2fit", self.g2fit)
        self.g2fit.fit_t_comp = self.g2fit.fit_t * self.g2fit.fit_pedestal / self.g2fit.fit_contrast
        
        E_ph = self.E_ph_box_used
        if E_ph == 0:
            self.g2_plot_idx = int(self.g2fit.omega.size / 2)
        else:
            self.g2_plot_idx = (numpy.abs(self.g2fit.omega * hr_eV_s - E_ph)).argmin()
        # except:
            # print('spectrum analysis: could not correlate or fit')
            # pass
        
        
        
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

        self.histogram_full = win.addPlot(row=1, col=0)
        self.histogram_full.setLabel('bottom', '<math>W/W<sub>mean</sub></math>')
        self.histogram_full.setLabel('left', 'full events', units='')
        self.histogram_full.clear()
        
        self.label_hist_full = pg.LabelItem(justify='right')
        win.addItem(self.label_hist_full, row=0, col=0)
        
        self.histogram_full_curve = self.histogram_full.plot([0,0], [0], stepMode=step_arg,  fillLevel=0,  brush=(100,100,100,100))
        
        self.histogram_full_fit_curve = self.histogram_full.plot(pen=pg.mkPen(color=(200, 0, 0), width=3))
        
    def plot_hist_full(self):
        if self.worth_plotting() and self.spar.events>2:
            spar = self.spar_screwed
            try:
                W, W_hist, W_bins = spar.calc_histogram(bins=self.hist_nbins, normed=True)
            except ValueError:
                W_bins = np.arange(11)
                W_hist = np.ones(10)
                W = np.ones(10)
            bin_width = W_bins[1]-W_bins[0]

            Wm = numpy.mean(W) #average power calculated
            # print("Wm_full", Wm)
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
            #self.histogram_full.clear()
            self.histogram_full_curve.setData(W_bins/Wm, W_hist*Wm*self.spar.events/self.hist_nbins)
            #self.histogram_full.plot(W_bins/Wm, W_hist, stepMode=True,  fillLevel=0,  brush=(100,100,100,100), clear=True)
             
            fit_p0 = [Wm, Wm**2 / numpy.mean((W - Wm)**2)]
            _, fit_p = fit_gamma_dist(W_bins[1:]-bin_width/2, W_hist, gamma_dist_function, fit_p0)
            Wm_fit, M_fit = fit_p # fit of average power and number of modes
            gama_dist = gamma_dist_function(W_bins[1:]-bin_width/2, Wm_fit, M_fit)*Wm*self.spar.events/self.hist_nbins
            gama_dist[gama_dist==np.inf]=np.nan
            #print('gama_dist_full=',gama_dist)
            self.histogram_full_fit_curve.setData((W_bins[1:]-bin_width/2)/Wm, gama_dist)
            
            self.label_hist_full.setText("<span style='font-size: 10pt', style='color: green'><math>M<sub>calc</sub></math>: %0.2f   <span style='color: red'><math>M<sub>fit</sub></math>: %0.2f</span>"%(M_calc, M_fit))
            

    def add_hist_peak_widget(self):
        # print('adding hist_peak widget')
        win = pg.GraphicsLayoutWidget()
        layout = QtGui.QGridLayout()
        self.ui.widget_histogram_peak.setLayout(layout)
        layout.addWidget(win)

        self.histogram_peak = win.addPlot(row=1, col=0)
        self.histogram_peak.setLabel('bottom', '<math>W/W<sub>mean</sub></math>')
        self.histogram_peak.setLabel('left', 'window events', units='')
        self.histogram_peak.clear()
        
        self.label_hist_peak = pg.LabelItem(justify='right')
        win.addItem(self.label_hist_peak, row=0, col=0)
        
        self.histogram_peak_curve = self.histogram_peak.plot([0,0], [0], stepMode=step_arg,  fillLevel=0,  brush=(100,100,100,100))
        
        self.histogram_peak_fit_curve = self.histogram_peak.plot(pen=pg.mkPen(color=(200, 0, 0), width=3))
        
    def plot_hist_peak(self):
        if self.worth_plotting() and self.spar.events>2:
            
            # E_ph_box = self.ui.analysis_Eph_box.value()
            # dE_ph_box = self.ui.analysis_dEph_box.value()
            # nbins = int(self.ui.analysis_nbins_box.value())
            
            # if E_ph_box == 0:
                # if self.spar.events < 5:
                    # return
                # else:
                    # specmean = np.mean(self.spar.spec, axis=1)
                    # E_ph_val = self.spar.phen[specmean.argmax()]
            # else:
                # E_ph_val = E_ph_box
            
            #self.histogram_peak.clear()
            spar = self.spar_screwed
            try:
                W, W_hist, W_bins = spar.calc_histogram(E=[self.E_ph_box_used-self.dE_ph_box, self.E_ph_box_used+self.dE_ph_box], bins=self.hist_nbins, normed=True)
            except ValueError:
                W_bins = np.arange(11)
                W_hist = np.ones(10)
                W = np.ones(10)
            bin_width = W_bins[1]-W_bins[0]
            Wm = numpy.mean(W) #average power calculated
            sigm2 = numpy.mean((W - Wm)**2) / Wm**2 #sigma square (power fluctuations)
            M_calc = 1 / sigm2 #calculated number of modes
            integ = np.sum(W)*bin_width
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
            gama_hist = W_hist*Wm*self.spar.events/self.hist_nbins
            self.histogram_peak_curve.setData(W_bins/Wm, gama_hist)
            
            fit_p0 = [Wm, Wm**2 / numpy.mean((W - Wm)**2)]
            _, fit_p = fit_gamma_dist(W_bins[1:]-bin_width/2, W_hist, gamma_dist_function, fit_p0)
            Wm_fit, M_fit = fit_p # fit of average power and number of modes
            gama_dist = gamma_dist_function(W_bins[1:]-bin_width/2, Wm_fit, M_fit)*Wm*self.spar.events/self.hist_nbins
            gama_dist[gama_dist==np.inf]=np.nan
            #print('gama_dist_peak=',gama_dist)
            self.histogram_peak_fit_curve.setData((W_bins[1:]-bin_width/2)/Wm, gama_dist)
            
            #self.histogram_peak.plot(W_bins/Wm, W_hist, stepMode=True,  fillLevel=0,  brush=(100,100,100,100), clear=True)
            self.label_hist_peak.setText("<span style='font-size: 10pt', style='color: green'><math>M<sub>calc</sub></math>: %0.2f   <span style='color: red'><math>M<sub>fit</sub></math>: %0.2f</span>"%(M_calc, M_fit))

    def add_durr_widget(self):
        win = pg.GraphicsLayoutWidget()
        layout = QtGui.QGridLayout()
        self.ui.widget_fit_pulse_dur.setLayout(layout)
        layout.addWidget(win)
        
        self.fit_pulse_dur = win.addPlot()
        legend = self.fit_pulse_dur.addLegend()
        # legend.setBrush((0,0,0,0))
        self.fit_pulse_dur.setLabel('bottom', '<math>E<sub>ph</sub></math>', units='eV')
        self.fit_pulse_dur.setLabel('left', 'group duration', units='s')
        self.durr_comp_curve = self.fit_pulse_dur.plot(symbolBrush='r', name='fit')
        self.durr_curve = self.fit_pulse_dur.plot(symbolSize = 3, symbolBrush=(50,0,0,50), pen = (50,0,0,0), name='uncorrected fit')
        self.durr0_curve = self.fit_pulse_dur.plot(pen='w')
        self.fit_pulse_dur.setXLink(self.img_spectrum)
        
        self.g2_phen_curve = self.fit_pulse_dur.plot(pen='g')
        
        # self.fit_pulse_dur.setYRange(0,2)
        
    def update_durr_plot(self):
        if len(self.g2fit.fit_t_comp) == 0 or self.n_last_correlated == 0:
            print('no fit data, skipping update_durr_plot')
            return
        idx = self.g2fit.fit_t_comp>0
        # print('idx(self.g2fit.fit_t_comp>0) = ',idx)
        self.durr_curve.setData(self.g2fit.omega[idx] * hr_eV_s, self.g2fit.fit_t[idx])
        self.durr_comp_curve.setData(self.g2fit.omega[idx] * hr_eV_s, self.g2fit.fit_t_comp[idx])
        self.durr0_curve.setData(self.g2fit.omega[idx] * hr_eV_s, np.zeros_like(self.g2fit.fit_t_comp[idx]))

        try:
            if len(self.g2fit.fit_t_comp)>0:
                maxdur = np.nanmax(self.g2fit.fit_t_comp[idx])
                centerphen = self.g2fit.omega[self.g2_plot_idx] * hr_eV_s
                # print(maxdur, centerphen)
                self.g2_phen_curve.setData([centerphen, centerphen], [0,maxdur])
        except ValueError:
            print('ValueError in update_durr_plot')
            pass
        # self.fit_pulse_dur.setLimits(yMin=0)
        # self.fit_pulse_dur.setLimits([0,2])
        
        # print('pedestal')
        # print(self.g2fit.fit_pedestal)
        # print('contrast')
        # print(self.g2fit.fit_contrast)
    
    def add_g2_line_widget(self):
        win = pg.GraphicsLayoutWidget()
        layout = QtGui.QGridLayout()
        self.ui.widget_corr_line.setLayout(layout)
        layout.addWidget(win)
        
        self.fit_g2_plot = win.addPlot()
        self.fit_g2_plot.addLegend()
        # label = pg.LabelItem(justify='left', row=0, col=0)
        # win.addItem(label)
        self.fit_g2_plot.clear()
        self.fit_g2_plot.setLabel('bottom', '<math>dE<sub>ph</sub></math>', units='eV')
        self.fit_g2_plot.setLabel('left', 'g<sub>2</sub>')
        self.g2_measured_curve = self.fit_g2_plot.plot(symbolBrush='b', name='data')
        self.g2_fit_curve = self.fit_g2_plot.plot(pen='r', name='fit')
        # self.fit_g2_plot.setXLink(self.img_spectrum)
        self.fit_g2_plot.setYRange(0.5,2.5)
        self.g2_1_curve = self.fit_g2_plot.plot(pen=(200,200,200,150), style=QtCore.Qt.DashLine)
        self.g2_2_curve = self.fit_g2_plot.plot(pen=(200,200,200,150), style=QtCore.Qt.DashLine)
        self.label_durr_widget = pg.LabelItem(justify='right')
        win.addItem(self.label_durr_widget, row=0, col=0)
        
    def update_g2_line_plot(self):
        if len(self.g2fit.fit_t) == 0 or self.n_last_correlated == 0:
            print('no fit data, skipping update_durr_plot')
            return
        
        self.g2_measured_curve.setData(self.g2fit.domega * hr_eV_s, self.g2fit.g2_measured[self.g2_plot_idx])
        self.g2_fit_curve.setData(self.g2fit.domega * hr_eV_s, self.g2fit.g2_fit[self.g2_plot_idx])
        self.g2_1_curve.setData(self.g2fit.domega * hr_eV_s, np.ones_like(self.g2fit.domega))
        self.g2_2_curve.setData(self.g2fit.domega * hr_eV_s, np.ones_like(self.g2fit.domega)*2)
        self.label_durr_widget.setText("<span style='font-size: 10pt', style='color: black'> <math>&lt;E<sub>ph</sub>&gt;</math> = %0.2f eV</span>"%(self.g2fit.omega[self.g2_plot_idx] * hr_eV_s))
        
    def add_rosa_widget(self):
        win = pg.GraphicsLayoutWidget()
        layout = QtGui.QGridLayout()
        self.ui.widget_rosa.setLayout(layout)
        layout.addWidget(win)
        
        self.rosa_plot = win.addPlot()
        self.add_rosaimage_item()
        
    def add_rosaimage_item(self):
        self.rosa_plot.clear()
        self.rosa_plot.setLabel('left', "E_ph", units='eV')
        self.rosa_plot.setLabel('bottom', "dt", units='s')
        self.img_rosa = pg.ImageItem()

        self.rosa_plot.addItem(self.img_rosa)

        colormap = cm.get_cmap('gist_earth_r') #"nipy_spectral")  # cm.get_cmap("CMRmap")
        colormap._init()
        lut = (colormap._lut * 255).view(np.ndarray)  # Convert matplotlib colormap from 0-1 to 0 -255 for Qt

        # Apply the colormap
        self.img_rosa.setLookupTable(lut)
        
    def update_rosa_plot(self):
    
        m_idx = self.reconstr.m_idx
        xaxis = self.reconstr.time_scale()[m_idx:]
        xydata = self.reconstr.recon.T[m_idx:,:]
        yaxis = self.reconstr.omega_bin * hr_eV_s
        
        # nx = 100
        # ny = 300
        # xaxis = np.linspace(0,10,nx)
        # yaxis = np.linspace(1000,1010,ny)
        # xydata = np.random.randn(nx,ny)
        scale_coef_xaxis = (xaxis.max() - xaxis.min())/len(xaxis)
        scale_coef_yaxis = (yaxis.max() - yaxis.min())/len(yaxis)
        
        self.add_rosaimage_item()
        self.img_rosa.setImage(xydata)
        self.img_rosa.scale(scale_coef_xaxis, scale_coef_yaxis)
        self.img_rosa.translate(xaxis[0]/scale_coef_xaxis, yaxis[0]/scale_coef_yaxis)
        
        
    def correlate_and_plot(self):
        if self.spar_screwed.events > 2:
            
            self.correlate()
            self.update_durr_plot()
            self.update_g2_line_plot()
            self.update_rosa_plot()
        else:
            print('not enough events for correlation')
        
    def correlate_and_plot_auto(self):
        n_new = self.n_events_processed - self.n_last_correlated
        correlate_every = self.ui.spinbox_correlate_every.value()
        if correlate_every > 0 and n_new > correlate_every:
            self.correlate_and_plot()
        
        
    def print_n_events(self):
        self.ui.label_30.setText("Process last {} of ".format(self.spar.events))
        
    def clear_all_plots(self):
        for widget in [self.img_spectrum, self.histogram_full, self.histogram_peak, self.fit_pulse_dur, self.fit_g2_plot]:
            widget.clear()
            
            
    def clear_all_curves(self):
        for curve in [self.spec_mean_curve, self.spec_last_curve, self.spec_max_curve, self.spec_min_curve, self.spec_window_l, self.spec_window_r,
        self.histogram_full_fit_curve, 
        self.histogram_peak_fit_curve, 
        self.durr_curve, self.durr_comp_curve,
        self.g2_measured_curve, self.g2_fit_curve]:
            curve.clear()
        self.histogram_full_curve.setData([0,0],[0])
        self.histogram_peak_curve.setData([0,0],[0])
