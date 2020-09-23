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
        
        self.doocs_address_label = '' #костыль
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
        
        min_val = sbin * (int(min(self.doocs_vals_hist) / sbin))
        max_val = max(self.doocs_vals_hist)
        # print('min_DOOCS_val', min_val)
        # print('max_DOOCS_val', max_val)
        # print('sbin', sbin)
        self.doocs_bins = np.arange(min_val, max_val, sbin)
        
        spec = np.array(self.spec_hist)
        # print('shape of spec array', spec.shape)
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

        if self.ui.pb_start.text() == "Start" or not self.ui.sb_corr_2d_run.isChecked():
            return
        
        self.phen = self.parent.x_axis
        
        # print('min_self.phen_val', min(self.phen))
        # print('max_self.phen_val', max(self.phen))
        
        self.spec_hist.append(self.parent.spectrum_event)
        
        #self.doocs_dev is None and
        if self.parent.ui.combo_hirex.currentText() == "DUMMY HIREX":
        
            dummy_val = np.sin(time.time()/10)*7.565432 + 25
            self.doocs_vals_hist.append(dummy_val) #fix
            # print('dummy hirex value', dummy_val)
            self.doocs_address_label = 'dummy label'
        else:
            self.doocs_vals_hist.append(self.doocs_dev.get_value())
            self.doocs_address_label = self.ui.le_doocs_ch_cor2d.text()
        
        n_shots = int(self.ui.sb_n_shots_2d.value())
        
        
        
        if len(self.spec_hist) > n_shots:
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
            
            self.add_image_item()
            self.img.setImage(self.spec_binned)
            
            #elegant but maybe wrong
            # rect = QtCore.QRect(min(self.doocs_bins), min(self.phen), max(self.doocs_bins)-min(self.doocs_bins), max(self.phen)-min(self.phen))
            # self.img.setRect(rect)
            self.img.scale(scale_xaxis, scale_yaxis)
            self.img.translate(translate_xaxis, translate_yaxis)
        

    def add_image_widget(self):
        win = pg.GraphicsLayoutWidget()
        layout = QtGui.QGridLayout()
        self.ui.widget_corr2d.setLayout(layout)
        layout.addWidget(win)

        self.img_plot = win.addPlot()
        self.add_image_item()
        
    def add_image_item(self):
        self.img_plot.clear()

        self.img_plot.setLabel('left', "E_ph", units='KeV')
        self.img_plot.setLabel('bottom', self.doocs_address_label, units='')

        self.img = pg.ImageItem()

        self.img_plot.addItem(self.img)

        colormap = cm.get_cmap('viridis') #"nipy_spectral")  # cm.get_cmap("CMRmap")
        colormap._init()
        lut = (colormap._lut * 255).view(np.ndarray)  # Convert matplotlib colormap from 0-1 to 0 -255 for Qt

        # Apply the colormap
        self.img.setLookupTable(lut)