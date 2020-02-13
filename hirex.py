from PyQt5.QtWidgets import QFrame, QMainWindow
import sys
import os
import argparse
import time
from scipy.signal import find_peaks

import numpy as np
import pyqtgraph as pg
from scipy.optimize import curve_fit
from threading import Thread, Event
path = os.path.realpath(__file__)
indx = path.find("hirex.py")
print("PATH to main file: " + os.path.realpath(__file__) + " path to folder: "+ path[:indx])
sys.path.insert(0, path[:indx])
from matplotlib import cm
from gui.spectr_gui import *
from mint.xfel_interface import *
from gui.settings_gui import *
from mint.devices import Spectrometer, BunchNumberCTRL
from scan import ScanInterface
from scipy import ndimage
import logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)





AVAILABLE_MACHINE_INTERFACES = [XFELMachineInterface, TestMachineInterface]

#HIREX_N_PIXELS = 1280
#DOOCS_CTRL_N_BUNCH = "XFEL.UTIL/BUNCH_PATTERN/CONTROL/NUM_BUNCHES_REQUESTED_2"
DIR_NAME = "hirex"


class Background(Thread):
    def __init__(self, mi, device):
        super(Background, self).__init__()
        self.mi = mi
        self.devmode = False
        self.device = device
        self.nshots = 100
        self.background = []
        self._stop_event = Event()

    def load(self):
        self.background = np.array([])
        try:
            self.background = np.loadtxt("background.txt")
        except Exception as ex:
            print("Problem with background: {}. Exception was: {}".format("background.txt", ex))

        return self.background
    
    def run(self):
        Y = []
        for i in range(self.nshots):
            x = self.device.get_value()
            if self.devmode:
                x = np.zeros_like(x) + 3 * np.exp(-np.linspace(-10, 10, num=len(x)) ** 2 / ((1 * 2)))
            Y.append(x)
            time.sleep(0.1)
        self.background = np.mean(Y, axis=0)
        np.savetxt("background.txt", self.background)
        print("Background finished")

    def stop(self):
        print("stop")
        self._stop_event.set()


class SpectrometerWindow(QMainWindow):
    """ Main class for the GUI application """
    def __init__(self):
        """
        Initialize the GUI and QT UI aspects of the application.
        Initialize the scan parameters.
        Connect start and logbook buttons on the scan panel.
        Initialize the plotting.
        Make the timer object that updates GUI on clock cycle during a scan.
        """
        # PATHS

        self.tool_args = None
        self.parse_arguments()
        self.dev_mode = self.tool_args.devmode

        args = vars(self.tool_args)
        if self.dev_mode:
            self.mi = TestMachineInterface(args)
        else:
            class_name = self.tool_args.mi
            #print(class_name)
            if class_name not in globals():
                print("Could not find Machine Interface with name: {}. Loading XFELMachineInterface instead.".format(class_name))
                self.mi = XFELMachineInterface(args)
            else:
                self.mi = globals()[class_name](args)


        self.path = path[:path.find(DIR_NAME)]
        self.config_dir = self.path + DIR_NAME + os.sep + "configs" + os.sep
        self.config_file = self.config_dir + "config.json"
        self.settings_file = self.config_dir + "settings.json"
        self.gui_dir = self.path + DIR_NAME + os.sep + "gui"+ os.sep

        # initialize
        QFrame.__init__(self)
        self.ui = MainWindow(self)

        self.settings = None
        #self.load_settings()
        try:
            self.load_settings()
        except:
            self.run_settings_window()
            self.settings.apply_settings()

        self.scantool = ScanInterface(parent=self)

        self.load_objects()

        self.add_plot()
        self.add_image_widget()
        self.ui.restore_state(self.config_file)

        self.actual_n_bunchs = self.bunch_num_ctrl.get_value()
        self.data_2d = np.random.rand(self.spectrometer.num_px, 1000)

        self.timer_live = pg.QtCore.QTimer()
        self.timer_live.timeout.connect(self.live_spec)
        
        
        self.ui.pb_start.clicked.connect(self.start_stop_live_spectrum)
        self.ui.pb_background.clicked.connect(self.take_background)
        self.spectrum_list = []
        self.ave_spectrum = []
        self.background = self.back_taker.load()
            
        self.ui.sb_ev_px.valueChanged.connect(self.calibrate_axis)
        self.ui.sb_E0.valueChanged.connect(self.calibrate_axis)
        self.ui.sb_px1.valueChanged.connect(self.calibrate_axis)
        
        #self.spectrometer.calibrate_axis(ev_px=self.ui.sb_ev_px.value(), E0=self.ui.sb_E0.value(), px1=self.ui.sb_px1.value())
        self.calibrate_axis()
        self.gauss_coeff_fit = None
        self.ui.pb_estim_px1.clicked.connect(self.fit_guass)
        self.ui.chb_show_fit.stateChanged.connect(self.show_fit)

        self.back_taker_status = pg.QtCore.QTimer()
        self.back_taker_status.timeout.connect(self.is_back_taker_alive)

        self.ui.actionSettings.triggered.connect(self.run_settings_window)
        self.ui.pb_cross_calib.clicked.connect(self.cross_calibrate)
        self.calib_energy_coef = 1
        self.add_corel_plot()
        self.plot1.scene().sigMouseMoved.connect(self.mouseMoved)
        #proxy = pg.SignalProxy(self.plot1.scene().sigMouseMoved, rateLimit=60, slot=self.mouseMoved)

    def load_objects(self):

        self.bunch_num_ctrl = BunchNumberCTRL(self.mi, self.doocs_ctrl_num_bunch)
        self.spectrometer = Spectrometer(self.mi, eid=self.hirex_doocs_ch)
        self.spectrometer.num_px = self.hrx_n_px
        self.spectrometer.devmode = self.dev_mode
        self.back_taker = Background(mi=self.mi, device=self.spectrometer)

    def get_transmission(self):
        value = self.ui.sb_transmission.value()
        if value == 0:
            value = 0.0000001
        return value

    def cross_calibrate(self):
        """
        Cross calibrate with spectrum with pulse energy

        :param spectrum: array
        :param transmission: transmission coefficient 0 - 1
        :param pulse_energy: in [uJ]
        :return: calibration coefficient
        """
        if len(self.ave_spectrum) < 3:
            return

        pulse_energy = self.mi.get_value(self.slow_xgm_signal)
        transmission = self.get_transmission()
        self.calib_energy_coef = self.spectrometer.cross_calibrate(self.ave_spectrum, transmission, pulse_energy)


    def run_settings_window(self):
        if self.settings is None:
            self.settings = HirexSettings(parent=self)
        self.settings.show()

    def fit_guass(self):
        if len(self.ave_spectrum) == 0:
            self.error_box("Press Start first")
            return
        mu = self.spectrometer.fit_guass(self.ave_spectrum)
        self.ui.sb_px1.setValue(mu)
        
        print("A, mu, sigma = ", self.spectrometer.gauss_coeff_fit)
    
    def show_fit(self):
        if self.spectrometer.gauss_coeff_fit is None:
            self.error_box("Estimate Px1 first")
            return

        def gauss(x, *p):
            A, mu, sigma = p
            return A*np.exp(-(x-mu)**2/(2.*sigma**2))

        if self.ui.chb_show_fit.isChecked():
        
            self.plot1.addItem(self.fit_func)
            self.plot1.setLabel('left', "A", units='au')
            self.plot1.setLabel('bottom', "", units='px')
            self.x_axis = np.arange(len(self.ave_spectrum))
            gauss_fit = gauss(self.x_axis, *self.spectrometer.gauss_coeff_fit)
            self.fit_func.setData(self.x_axis, gauss_fit)
            #self.plot1.enableAutoScale()
            self.plot1.enableAutoRange()
        else:
            self.calibrate_axis()
            self.plot1.removeItem(self.fit_func)
            self.plot1.legend.removeItem(self.fit_func.name())
            self.plot1.setLabel('left', "A", units='au')
            self.plot1.setLabel('bottom', "", units='eV')
            #self.plot1.enableAutoScale()
            self.plot1.enableAutoRange()

    def calibrate_axis(self):
        ev_px = self.ui.sb_ev_px.value()
        E0 = self.ui.sb_E0.value()
        px1 = self.ui.sb_px1.value()
        self.x_axis = self.spectrometer.calibrate_axis(ev_px, E0, px1)


    def is_back_taker_alive(self):
        """
        Method to check if the ResponseMatrixCalculator thread is alive.
        it is needed to change name and color of the pushBatton pb_calc_RM.
        When RMs caclulation is finished. If the thread is dead QTimer self.rm_calc is stopped
        :return:
        """
        if not self.back_taker.is_alive():
            self.ui.pb_background.setStyleSheet("color: rgb(85, 255, 255);")
            self.ui.pb_background.setText("Take Background")
            self.background = self.back_taker.background
            self.back_taker_status.stop()
            self.bunch_num_ctrl.set_value(self.actual_n_bunchs)

    def take_background(self):
    
        if self.ui.pb_background.text() == "Taking ...              ":
            self.ui.pb_background.setStyleSheet("color: rgb(85, 255, 255);")
            self.ui.pb_background.setText("Take Background")
            if self.back_taker.is_alive():
                self.back_taker.stop()
                self.bunch_num_ctrl.set_value(self.actual_n_bunchs)
        else:
            if self.ui.pb_start.text() == "Start":
                self.error_box("Start HIREX first")
                return
            self.actual_n_bunchs = self.bunch_num_ctrl.get_value()
            self.back_taker = Background(mi=self.mi, device=self.spectrometer)
            self.back_taker.devmode = self.dev_mode
            self.bunch_num_ctrl.set_value(0)
            time.sleep(0.15)
            self.back_taker.nshots = int(self.ui.sb_nbunch_back.value())
            self.back_taker.doocs_channel = str(self.ui.le_a.text())
            if not self.back_taker.is_alive():
                self.back_taker.start()
                self.back_taker_status.start()
            self.ui.pb_background.setText("Taking ...              ")
            self.ui.pb_background.setStyleSheet("color: rgb(85, 255, 127);")


    def live_spec(self):

        spectrum = self.spectrometer.get_value()
        if self.ui.chb_a.isChecked():
            if len(self.background) != len(spectrum):
                self.error_box("Take Background")
                self.ui.chb_a.setChecked(False)
            else:
                spectrum -= self.background
        self.spectrum_list.insert(0, spectrum)
        self.ave_spectrum = np.mean(self.spectrum_list, axis=0)

        n_av = int(self.ui.sb_av_nbunch.value())
        if len(self.spectrum_list) > n_av:
            self.spectrum_list = self.spectrum_list[:n_av]

        filtr_av_spectrum = ndimage.gaussian_filter(self.ave_spectrum, sigma=self.ui.sb_gauss_filter.value())
        peaks, _ = find_peaks(filtr_av_spectrum,  distance=self.ui.sb_mkn_dist_peaks.value(),
                               height=np.max(filtr_av_spectrum)*self.ui.sb_low_thresh.value()/100.)
        self.peak_ev_list = self.x_axis[peaks]

        single_integr = np.trapz(spectrum, self.x_axis)/self.get_transmission() * self.calib_energy_coef
        ave_integ = np.trapz(self.ave_spectrum, self.x_axis) / self.get_transmission() * self.calib_energy_coef

        self.peak_ev = self.x_axis[np.argmax(self.ave_spectrum)]


        self.single.setData(self.x_axis, spectrum)
        self.average.setData(x=self.x_axis, y=self.ave_spectrum)
        #self.average.setData(x=self.x_axis, y=filtr_av_spectrum)
        
        self.data_2d = np.roll(self.data_2d, 1, axis=1)

        self.data_2d[:, 0] = spectrum# single_sig_wo_noise

        self.img.setImage(self.data_2d[ self.img_idx1:self.img_idx2])
        #if not self.is_txt_item:
        #    self.plot1.addItem(self.textItem)
        #    self.is_txt_item = True
        #self.update_text("Av = " + str(np.round(ave_integ, 1)) + " uJ \nEpk = " + str(np.round(self.peak_ev, 1)) + "eV")

        self.label2.setText(
        "<span style='font-size: 16pt', style='color: yellow'>%0.1f &mu;J   <span style='color: yellow'> @ %0.1f eV</span>" % (
            ave_integ, self.peak_ev))

    def start_stop_live_spectrum(self):
        if self.ui.pb_start.text() == "Stop":
            self.timer_live.stop()
            self.ui.pb_start.setStyleSheet("color: rgb(255, 0, 0);")
            self.ui.pb_start.setText("Start")
        else:


            self.timer_live.start(100)
            self.ui.pb_start.setText("Stop")
            self.ui.pb_start.setStyleSheet("color: rgb(85, 255, 127);")
            px1 = int(self.ui.sb_px1.value())
            img_idx1 = int(px1 - 250)
            img_idx2 = int(px1 + 250)
            self.img_idx1 = img_idx1 if img_idx1 >= 0 else 0
            self.img_idx2 = img_idx2 if img_idx2 < self.spectrometer.num_px else -1
            scale_coef_xaxis = (self.x_axis[self.img_idx2] - self.x_axis[self.img_idx1]) / (
                        self.img_idx2 - self.img_idx1)
            translate_coef_xaxis = self.x_axis[self.img_idx1] / scale_coef_xaxis

            self.add_image_item()

            self.img.scale(scale_coef_xaxis, 1)
            self.img.translate(translate_coef_xaxis, 0)
            #self.plot1.addItem(self.textItem)

    def update_text(self, text=None):
        x_left = self.plot1.viewRange()[0][0]
        x_right = self.plot1.viewRange()[0][1]
        y_down = self.plot1.viewRange()[1][0]
        y_up = self.plot1.viewRange()[1][1]
        x = x_left + (x_right - x_left)*0.7
        y = y_down + (y_up - y_down)*0.9

        #print(self.plot1.viewRange())
        self.textItem.setText(text)
        self.textItem.setPos(x, y)

    def closeEvent(self, event):
        #if self.orbit.adaptive_feedback is not None:
        #    self.orbit.adaptive_feedback.close()c
        if 1:
            self.ui.save_state(self.config_file)
        logger.info("close")
        event.accept()  # let the window close

    def parse_arguments(self):
        parser = argparse.ArgumentParser(description="Ocelot Orbit Correction",
                                         add_help=False)
        parser.set_defaults(mi='XFELMachineInterface')
        parser.add_argument('--devmode', action='store_true',
                            help='Enable development mode.', default=False)

        parser_mi = argparse.ArgumentParser()

        mis = [mi.__class__.__name__ for mi in AVAILABLE_MACHINE_INTERFACES]
        subparser = parser_mi.add_subparsers(title='Machine Interface Options', dest="mi")
        for mi in AVAILABLE_MACHINE_INTERFACES:
            mi_parser = subparser.add_parser(mi.__name__, help='{} arguments'.format(mi.__name__))
            mi.add_args(mi_parser)

        self.tool_args, others = parser.parse_known_args()

        if len(others) != 0:
            self.tool_args = parser_mi.parse_args(others, namespace=self.tool_args)


    def add_plot(self):

        win = pg.GraphicsLayoutWidget()
        #justify='right',,
        self.label = pg.LabelItem(justify='left', row=0, col=0)
        win.addItem(self.label)


        #self.plot1 = win.addPlot(row=0, col=0)
        self.plot1 = win.addPlot(row=1, col=0)

        self.label2 = pg.LabelItem( justify='right')
        win.addItem(self.label2, row=0, col=0)

        self.plot1.setLabel('left', "A", units='au')
        self.plot1.setLabel('bottom', "", units='eV')

        self.plot1.showGrid(1, 1, 1)

        self.plot1.getAxis('left').enableAutoSIPrefix(enable=False)  # stop the auto unit scaling on y axes
        layout = QtGui.QGridLayout()
        self.ui.widget.setLayout(layout)
        layout.addWidget(win, 0, 0)

        self.plot1.setAutoVisible(y=True)

        self.plot1.addLegend()
        color = QtGui.QColor(0, 255, 255)
        pen = pg.mkPen(color, width=2)
        self.single = pg.PlotCurveItem(name='single')

        self.plot1.addItem(self.single)

        color = QtGui.QColor(255, 0, 0)
        # pen = pg.mkPen((255, 0, 0), width=2)
        pen = pg.mkPen((51, 255, 51), width=2)
        #self.average = pg.PlotCurveItem(x=[], y=[], pen=pen, name='average')
        self.average = pg.PlotCurveItem( pen=pen, name='average')

        self.plot1.addItem(self.average)
        
        pen = pg.mkPen((0, 255, 255), width=2)

        self.fit_func = pg.PlotCurveItem(pen=pen, name='Gauss Fit')

        #self.plot1.addItem(self.fit_func)
        #self.plot1.enableAutoRange(False)
        #self.textItem = pg.TextItem(text="", border='w', fill=(0, 0, 0))
        # self.textItem.setPos(10, 10)

        # cross hair
        self.vLine = pg.InfiniteLine(angle=90, movable=False)
        self.hLine = pg.InfiniteLine(angle=0, movable=False)
        self.plot1.addItem(self.vLine, ignoreBounds=True)
        self.plot1.addItem(self.hLine, ignoreBounds=True)

        #self.plot1.sigRangeChanged.connect(self.zoom_signal)

    def mouseMoved(self, evt):
        #print("here", evt)
        #pos = evt.x(), evt.y() #evt[0]  ## using signal proxy turns original arguments into a tuple
        #print(evt.x())
        if self.plot1.sceneBoundingRect().contains(evt.x(), evt.y()):
            mousePoint = self.plot1.vb.mapSceneToView(evt)
            # index = int(mousePoint.x())
            array = np.asarray(self.x_axis)
            index = (np.abs(array - mousePoint.x())).argmin()
            #print(mousePoint.x(), index, len(self.x_axis))
            if index > 0 and index < len(self.x_axis):
                self.label.setText(
                    "<span style='font-size: 16pt', style='color: green'>x=%0.1f,   <span style='color: red'>y=%0.1f</span>" % (
                    mousePoint.x(), mousePoint.y()))
            self.vLine.setPos(mousePoint.x())
            self.hLine.setPos(mousePoint.y())

    def add_image_widget(self):
        win = pg.GraphicsLayoutWidget()
        layout = QtGui.QGridLayout()
        self.ui.widget_2.setLayout(layout)
        layout.addWidget(win)

        self.img_plot = win.addPlot()
        self.add_image_item()

    def add_image_item(self):
        self.img_plot.clear()

        self.img_plot.setLabel('left', "N bunch", units='')
        self.img_plot.setLabel('bottom', "", units='eV')

        self.img = pg.ImageItem()

        self.img_plot.addItem(self.img)

        colormap = cm.get_cmap('viridis') #"nipy_spectral")  # cm.get_cmap("CMRmap")
        colormap._init()
        lut = (colormap._lut * 255).view(np.ndarray)  # Convert matplotlib colormap from 0-1 to 0 -255 for Qt

        # Apply the colormap
        self.img.setLookupTable(lut)

    def add_corel_plot(self):

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
        self.single_scatter = pg.ScatterPlotItem(name='single')

        self.plot_cor.addItem(self.single_scatter)

        pen = pg.mkPen((255, 0, 0), width=2)
        # self.average = pg.PlotCurveItem(x=[], y=[], pen=pen, name='average')
        self.average_scatter = pg.ScatterPlotItem(pen=pen, name='average')

        self.plot_cor.addItem(self.average_scatter)

        pen = pg.mkPen((0, 255, 255), width=2)

        self.fit_func_scatteer = pg.PlotCurveItem(pen=pen, name='Gauss Fit')

        # self.plot1.addItem(self.fit_func)
        self.plot_cor.enableAutoRange(False)
        #self.textItem = pg.TextItem(text="", border='w', fill=(0, 0, 0))


    def error_box(self, message):
        QtGui.QMessageBox.about(self, "Error box", message)

    def question_box(self, message):
        #QtGui.QMessageBox.question(self, "Question box", message)
        reply = QtGui.QMessageBox.question(self, "Recalculate ORM?",
                "Recalculate Orbit Response Matrix?",
                QtGui.QMessageBox.Yes | QtGui.QMessageBox.No)
        if reply == QtGui.QMessageBox.Yes:
            return True

        return False

    def zoom_signal(self):
        #s = self.plot1.viewRange()[0][0]

        s_up = self.plot1.viewRange()[0][0]
        s_down = self.plot1.viewRange()[0][1]
        #print(np.argwhere(self.x_axis > s_up))
        #print(np.argwhere(self.x_axis < s_down))
        #print(f"s_down = {s_down}, s_up = {s_up}, min(axis) = {np.min(self.x_axis)}, max(axis) = {np.max(self.x_axis)}")
        #indx1 = np.argwhere(self.x_axis > s_up)[0]
        #indx2 = np.argwhere(self.x_axis < s_down)[-1]

        #print(s_up, s_down, indx1, indx2)
        #s_up = s_up if s_up <= s_pos[-1] else s_pos[-1]
        #s_down = s_down if s_down >= s_pos[0] else s_pos[0]

    def load_settings(self):
        logger.debug("load settings ... ")
        with open(self.settings_file, 'r') as f:
            table = json.load(f)

        self.hirex_doocs_ch = table["le_hirex_ch"]
        self.transmission__doocs_ch = table["le_trans_ch"]
        self.hrx_n_px = table["sb_hrx_npx"]
        self.logbook = table["logbook"]
        self.doocs_ctrl_num_bunch = table["le_ctrl_num_bunch"]
        self.fast_xgm_signal = table["le_fast_xgm"]
        self.slow_xgm_signal = table["le_slow_xgm"]
        if "server" in table.keys():
            self.server = table["server"]
        else:
            self.server = "XFEL"

        logger.debug("load settings ... OK")

    def loadStyleSheet(self):
        """ Sets the dark GUI theme from a css file."""
        try:
            self.cssfile = "gui/style.css"
            with open(self.cssfile, "r") as f:
                self.setStyleSheet(f.read())
        except IOError:
            logger.error('No style sheet found!')


def main():


    #make pyqt threadsafe
    QtCore.QCoreApplication.setAttribute(QtCore.Qt.AA_X11InitThreads)
    #create the application
    app = QApplication(sys.argv)


    window = SpectrometerWindow()


    #show app
    #window.setWindowIcon(QtGui.QIcon('gui/angry_manul.png'))
    # setting the path variable for icon
    #path = os.path.join(os.path.dirname(sys.modules[__name__].__file__), 'gui/manul.png')
    #app.setWindowIcon(QtGui.QIcon(path))
    window.show()
    window.raise_()
    #Build documentaiton if source files have changed
    # TODO: make more universal
    #os.system("cd ./docs && xterm -T 'Ocelot Doc Builder' -e 'bash checkDocBuild.sh' &")
    #exit script
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()