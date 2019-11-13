from PyQt5.QtWidgets import QFrame, QMainWindow
import sys
import os
import argparse
import time

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
import logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)





AVAILABLE_MACHINE_INTERFACES = [XFELMachineInterface, TestMachineInterface]

#HIREX_N_PIXELS = 1280
#DOOCS_CTRL_N_BUNCH = "XFEL.UTIL/BUNCH_PATTERN/CONTROL/NUM_BUNCHES_REQUESTED_2"
DIR_NAME = "hirex"

class BunchCTRL():
    def __init__(self, mi, doocs_ch):
        self.mi = mi
        self.doocs_ch = doocs_ch

    def get_value(self):
        val = self.mi.get_value(self.doocs_ch)
        return val

    def set_value(self, num):
        self.mi.set_value(self.doocs_ch, num)



class Background(Thread):
    def __init__(self, mi):
        super(Background, self).__init__()
        self.mi = mi
        self.doocs_channel = None
        self.nshots = 100
        self.noise = None
        self._stop_event = Event()

    def load(self):
        self.noise = np.array([])
        try:
            self.noise = np.loadtxt("background.txt")
        except Exception as ex:
            print("Problem with background: {}. Exception was: {}".format("background.txt", ex))
        return self.noise
    
    def run(self):
        Y = []
        for i in range(self.nshots):
            x = self.mi.get_value(self.doocs_channel)
            Y.append(x)
            time.sleep(0.1)
        self.noise = np.mean(Y, axis=0)
        np.savetxt("background.txt", self.noise)
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

        self.back_taker = Background(mi=self.mi)
        #print(self.mi.__class__)
        #path = os.path.realpath(__file__)
        indx = path.find("hirex.py")
        self.path2main_folder = path[:indx]
        self.path2gui = self.path2main_folder + "gui" +  os.sep

        self.set_file = "settings.json"
        # initialize
        QFrame.__init__(self)
        self.ui = MainWindow(self)

        self.path = path[:path.find(DIR_NAME)]
        self.config_dir = self.path + DIR_NAME + os.sep + "configs" + os.sep
        self.config_file = self.config_dir + "settings.json"
        self.gui_dir = self.path + DIR_NAME + os.sep + "gui"+ os.sep

        self.settings = None
        try:
            self.load_settings()
        except:
            self.run_settings_window()
            self.settings.apply_settings()
        self.bunch_ctrl = BunchCTRL(self.mi, self.doocs_ctrl_num_bunch)

        self.add_plot()
        self.add_image_widget()
        self.ui.restore_state(self.set_file)

        self.actual_n_bunchs = self.bunch_ctrl.get_value()
        self.data_2d = np.random.rand(self.hrx_n_px, 1000)

        self.timer_live = pg.QtCore.QTimer()
        self.timer_live.timeout.connect(self.live_spec)
        
        
        self.ui.pb_start.clicked.connect(self.start_stop_live_spectrum)
        self.ui.pb_background.clicked.connect(self.take_background)
        self.X = []
        self.av_sig = []
        self.noise = self.back_taker.load()
            
        self.ui.sb_ev_px.valueChanged.connect(self.calibration)
        self.ui.sb_E0.valueChanged.connect(self.calibration)
        self.ui.sb_px1.valueChanged.connect(self.calibration)
        
        self.calibration()
        self.gauss_coeff_fit = None
        self.ui.pb_estim_px1.clicked.connect(self.fit_guass)
        self.ui.chb_show_fit.stateChanged.connect(self.show_fit)

        self.back_taker_status = pg.QtCore.QTimer()
        self.back_taker_status.timeout.connect(self.is_back_taker_alive)

        self.ui.actionSettings.triggered.connect(self.run_settings_window)

    def run_settings_window(self):
        if self.settings is None:
            self.settings = HirexSettings(parent=self)
        self.settings.show()

    def fit_guass(self):
        if len(self.av_sig) == 0:
            return 
        y = self.av_sig
        x = np.arange(len(y))
        def gauss(x, *p):
            A, mu, sigma = p
            return A*np.exp(-(x-mu)**2/(2.*sigma**2))

        # (A, mu, sigma)
        p0 = [np.max(y), np.argmax(y), 30]

        self.gauss_coeff_fit, var_matrix = curve_fit(gauss, x, y, p0=p0)
        mu = self.gauss_coeff_fit[1]
        self.ui.sb_px1.setValue(mu)
        
        print("A, mu, sigma = ", self.gauss_coeff_fit)
    
    def show_fit(self):
        if self.gauss_coeff_fit is None:
            return

        def gauss(x, *p):
            A, mu, sigma = p
            return A*np.exp(-(x-mu)**2/(2.*sigma**2))

        if self.ui.chb_show_fit.isChecked():
        
            self.plot1.addItem(self.fit_func)
            self.plot1.setLabel('left', "A", units='au')
            self.plot1.setLabel('bottom', "", units='px')
            self.x_axis = np.arange(len(self.av_sig))
            gauss_fit = gauss(self.x_axis, *self.gauss_coeff_fit)
            self.fit_func.setData(self.x_axis, gauss_fit)
            #self.plot1.enableAutoScale()
            self.plot1.enableAutoRange()
        else:
            self.calibration()
            self.plot1.removeItem(self.fit_func)
            self.plot1.legend.removeItem(self.fit_func.name())
            self.plot1.setLabel('left', "A", units='au')
            self.plot1.setLabel('bottom', "", units='eV')
            #self.plot1.enableAutoScale()
            self.plot1.enableAutoRange()

    
    def calibration(self):
        ev_px = self.ui.sb_ev_px.value()
        E0 = self.ui.sb_E0.value()
        px1 = self.ui.sb_px1.value()
        start = E0 - px1*ev_px
        stop = E0 + (self.hrx_n_px - px1) * ev_px
        self.x_axis = np.linspace(start, stop, num=self.hrx_n_px)

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
            self.noise = self.back_taker.noise
            self.back_taker_status.stop()
            self.bunch_ctrl.set_value(self.actual_n_bunchs)

    def take_background(self):
    
        if self.ui.pb_background.text() == "Taking ...              ":
            self.ui.pb_background.setStyleSheet("color: rgb(85, 255, 255);")
            self.ui.pb_background.setText("Take Background")
            if self.back_taker.is_alive():
                self.back_taker.stop()
                self.bunch_ctrl.set_value(self.actual_n_bunchs)
        else:
            if self.ui.pb_start.text() == "Start":
                self.error_box("Start HIREX first")
                return
            self.actual_n_bunchs = self.bunch_ctrl.get_value()
            self.back_taker = Background(mi=self.mi)
            self.bunch_ctrl.set_value(0)
            time.sleep(0.15)
            self.back_taker.nshots = int(self.ui.sb_nbunch_back.value())
            self.back_taker.doocs_channel = str(self.ui.le_a.text())
            if not self.back_taker.is_alive():
                self.back_taker.start()
                self.back_taker_status.start()
            self.ui.pb_background.setText("Taking ...              ")
            self.ui.pb_background.setStyleSheet("color: rgb(85, 255, 127);")


    def live_spec(self):
        x = self.mi.get_value(str(self.ui.le_a.text()))
        self.X.insert(0, x)
        y = np.mean(self.X, axis=0) # np.random.rand(500)
        s = np.arange(len(x))
        single_sig_wo_noise = x - self.noise
        av_sig_wo_noise = y - self.noise
        if self.ui.chb_a.isChecked():
            single_sig = single_sig_wo_noise
            self.av_sig = av_sig_wo_noise
        else:
            single_sig = x
            self.av_sig = y 
            
        self.single.setData(self.x_axis, single_sig)
        self.average.setData(x=self.x_axis, y= self.av_sig)
        n_av = int(self.ui.sb_av_nbunch.value())
        if len(self.X) > n_av:
            self.X = self.X[:n_av]
        
        self.data_2d = np.roll(self.data_2d, 1, axis=1)

        self.data_2d[:, 0] = x# single_sig_wo_noise


        self.img.setImage(self.data_2d[self.img_idx1:self.img_idx2])

    def start_stop_live_spectrum(self):
        if self.ui.pb_start.text() == "Stop":
            self.timer_live.stop()
            self.ui.pb_start.setStyleSheet("color: rgb(85, 255, 255);")
            self.ui.pb_start.setText("Start")
        else:
            self.timer_live.start(100)
            self.ui.pb_start.setText("Stop")
            self.ui.pb_start.setStyleSheet("color: rgb(85, 255, 127);")
            px1 = int(self.ui.sb_px1.value())
            img_idx1 = int(px1 - 250)
            img_idx2 = int(px1 + 250)
            self.img_idx1 = img_idx1 if img_idx1 >= 0 else 0
            self.img_idx2 = img_idx2 if img_idx2 < self.hrx_n_px else -1
            scale_coef_xaxis = (self.x_axis[self.img_idx2] - self.x_axis[self.img_idx1]) / (
                        self.img_idx2 - self.img_idx1)
            translate_coef_xaxis = self.x_axis[self.img_idx1] / scale_coef_xaxis

            self.add_image_item()

            self.img.scale(scale_coef_xaxis, 1)
            self.img.translate(translate_coef_xaxis, 0)

    def closeEvent(self, event):
        #if self.orbit.adaptive_feedback is not None:
        #    self.orbit.adaptive_feedback.close()c
        if 1:
            self.ui.save_state(self.set_file)
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

        self.plot1 = win.addPlot(row=0, col=0)
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
        pen = pg.mkPen((255, 0, 0), width=2)
        #self.average = pg.PlotCurveItem(x=[], y=[], pen=pen, name='average')
        self.average = pg.PlotCurveItem( pen=pen, name='average')

        self.plot1.addItem(self.average)
        
        pen = pg.mkPen((0, 255, 255), width=2)

        self.fit_func = pg.PlotCurveItem(pen=pen, name='Gauss Fit')

        #self.plot1.addItem(self.fit_func)
        
        self.plot1.sigRangeChanged.connect(self.zoom_signal)

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
        with open(self.config_file, 'r') as f:
            table = json.load(f)

        self.hirex_doocs_ch = table["le_hirex_ch"]
        self.transmission__doocs_ch = table["le_trans_ch"]
        self.hrx_n_px = table["sb_hrx_npx"]
        self.logbook = table["logbook"]
        self.doocs_ctrl_num_bunch = table["le_ctrl_num_bunch"]

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