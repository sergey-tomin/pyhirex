import numpy as np
import pyqtgraph as pg
from threading import Thread, Event
from PyQt5 import QtGui, QtCore
import time
from mint.opt_objects import Device
from scipy import ndimage
from matplotlib import cm


class Crystal(Device):
    def __init__(self, mi, doocs_stop, doocs_start, doocs_target, doocs_speed, doocs_actual):
        self.mi = mi
        self.doocs_stop = doocs_stop
        self.doocs_target = doocs_target
        self.doocs_speed = doocs_speed
        self.doocs_start = doocs_start
        self.doocs_actual = doocs_actual
    
    def stop(self):
        print("STOP CRYSTAL")
        self.mi.set_value(self.doocs_stop, 1)
    
    def start(self):
        print("START")
        if self.is_busy():
            print("BUSY")
        else:
            self.mi.set_value(self.doocs_start, 1)

    def get_actual_value(self):
        x = self.mi.get_value(self.doocs_actual)
        return x
    
    def set_speed(self, val):
        print("SET speed: ", val)
        self.mi.set_value(self.doocs_speed, val)
    
    def set_value(self, val):
        print("SET val: ", val)
        self.mi.set_value(self.doocs_target, val)
    
    def in_possition(self):
        pass
    
    def is_busy(self):
        parts = self.doocs_actual.split("/")
        tmp = parts[:-1]
        tmp.append("HW_STATE")
        ch = "/".join(tmp)
        #ch = "XFEL.FEL/UNDULATOR.SASE2/MONOPA.2307.SA2/HW_STATE"
        #print("busy chennel", ch_re, ch, ch == ch_re)
        
        x = int(self.mi.get_value(ch))
        if x & 0x04 == 0:
            print("Crystal not busy", x)
            return False
        else:
            print("Crystal busy", x)
            return True
        


class ScanTool(Thread):
    def __init__(self, mi, crystal):
        super(ScanTool, self).__init__()
        self.mi = mi
        self.parent = None
        self.devmode = False
        self.crystal = crystal
        self.cont_mode = True
        self.timeout = 2
        self.val_range = []
        self.kill = False
        self._stop_event = Event()
        self.doocs_vals = []
        self.spectrums = []
        self.peak_list_vals = []

    def continues_scan(self):
        self.spectrums = np.empty([1, self.parent.hrx_n_px])
        self.crystal.set_speed(self.parent.ui.sb_speed.value())
        self.crystal.set_value(self.parent.ui.sb_target.value())
        time.sleep(0.5)
        self.crystal.start()
        
        #self.crystal.is_busy()
        time.sleep(0.1)
        while (not self.kill and self.crystal.is_busy()):
            x = self.crystal.get_actual_value()
            self.doocs_vals.append(x)
            self.spectrums = np.append(self.spectrums, self.parent.ave_spectrum.reshape(1, -1), axis=0)
            self.peak_list_vals.append(self.parent.peak_ev_list)
            time.sleep(self.timeout)
            print("busy", self.crystal.is_busy())
            
        self.crystal.stop()
        print("EXIT from the loop" )

    def step_scan(self):
        self.spectrums = np.empty([1, self.parent.hrx_n_px])
        for val in self.val_range:
            if self.kill:
                return
            # self.device.set_value(val)
            time.sleep(self.timeout)
            self.doocs_vals.append(val)
            self.peak_list_vals.append(self.parent.peak_ev_list)
            self.spectrums = np.append(self.spectrums, self.parent.ave_spectrum.reshape(1, -1), axis=0)


    def run(self):
        if self.cont_mode:
            self.continues_scan()
        else:
            self.step_scan()

        print("Scanning finished")

    def stop(self):
        print("stop")
        self._stop_event.set()


class ScanInterface:
    """
    Main class for orbit correction
    """
    def __init__(self, parent):
        self.parent = parent
        self.ui = self.parent.ui
        self.mi = self.parent.mi

        self.add_plot()
        self.add_image_widget()
        self.plot1.scene().sigMouseMoved.connect(self.mouseMoved)

        self.scanning = None
        self.plot_timer = pg.QtCore.QTimer()
        self.plot_timer.timeout.connect(self.plot_scan)

        self.watch_dog_timer = pg.QtCore.QTimer()
        self.watch_dog_timer.timeout.connect(self.check_scanning)
        self.watch_dog_timer.start(500)

        self.ui.pb_start_scan.clicked.connect(self.start_stop_scan)
        self.ui.pb_check_range.clicked.connect(self.check_range)
        self.ui.pb_show_map.clicked.connect(self.show_hide_map)
        
        self.ui.cb_crystal_select.addItem("Pitch Angle")
        self.doocs_start_crystal = None
        self.doocs_stop_crystal = None
        self.actual_angle = None
        self.doocs_target = None
        self.doocs_speed = None
        self.crystal = None
        self.get_properties()
        
    
    def get_properties(self):
        if 1:
            doocs_actual_angle = "XFEL.FEL/UNDULATOR.SASE2/MONOPA.2307.SA2/ANGLE"
            doocs_min_angle = "XFEL.FEL/UNDULATOR.SASE2/MONOPA.2307.SA2/MINANGLE"
            doocs_max_angle = "XFEL.FEL/UNDULATOR.SASE2/MONOPA.2307.SA2/MAXANGLE"
            doocs_speed = "XFEL.FEL/UNDULATOR.SASE2/MONOPA.2307.SA2/SPEED.SET"
            doocs_start_crystal = "XFEL.FEL/UNDULATOR.SASE2/MONOPA.2307.SA2/CONTROL.START XFEL.FEL/UNDULATOR.SASE2/MONOPA.2307.SA2/CONTROL.START"
            doocs_stop_crystal = "XFEL.FEL/UNDULATOR.SASE2/MONOPA.2307.SA2/CONTROL.STOP XFEL.FEL/UNDULATOR.SASE2/MONOPA.2307.SA2/CONTROL.STOP"
            doocs_target = "XFEL.FEL/UNDULATOR.SASE2/MONOPA.2307.SA2/ANGLE.SET"
            
            self.crystal = Crystal(mi=self.mi, 
                                   doocs_stop=doocs_stop_crystal,
                                   doocs_start=doocs_start_crystal, 
                                   doocs_target=doocs_target, 
                                   doocs_speed=doocs_speed, 
                                   doocs_actual=doocs_actual_angle)
        try:
            min_angle = self.mi.get_value(doocs_min_angle)
            max_angle = self.mi.get_value(doocs_max_angle)
            actual_angle = self.crystal.get_actual_value()
        except:
            self.crystal = None
            return 
        self.ui.sb_target.setMinimum(min_angle)
        self.ui.sb_target.setMaximum(max_angle)
        self.ui.sb_target.setValue(actual_angle)


    def check_scanning(self):
        if self.scanning is not None and not self.scanning.isAlive():
            self.ui.pb_start_scan.setStyleSheet("color: rgb(255, 0, 0);")
            self.ui.pb_start_scan.setText("Start")

    def check_range(self):
        str_range = str(self.ui.le_scan_range.text())
        try:
            x = eval(str_range)
            self.parent.error_box(str(x))
        except:
            self.parent.error_box("incorrect range")
            return

    def image_scale(self):

        scale_coef_xaxis = (self.parent.x_axis[-1] - self.parent.x_axis[0]) / (
                        len(self.parent.x_axis))
        translate_coef_xaxis = self.parent.x_axis[0] / scale_coef_xaxis


        self.img.scale(1, scale_coef_xaxis)
        self.img.translate(0, translate_coef_xaxis)

    def plot_scan(self):
        x = self.scanning.doocs_vals
        peaks = self.scanning.peak_list_vals
        #print("x = ", x)
        #print("peaks = ", peaks)


        # scale_coef_xaxis = (x[0] - x[-1]) / len(x)
        # translate_coef_xaxis = x[0] / scale_coef_xaxis
        # self.add_image_item()
        # self.img.scale(scale_coef_xaxis, 1)
        # self.img.translate(translate_coef_xaxis, 0)
        # print(np.shape(self.scanning.spectrums))
        # print(self.scanning.spectrums)
        self.img.setImage(self.scanning.spectrums)

        for i in range(self.ui.sb_num_peaks.value()):
            if len(peaks) > 0 and len(peaks[0]) > i:
                y = [peaks[i] for peaks in peaks]
                if len(x) == len(y):
                    self.peak_lines[i].setData(x, y)
            else:
                self.peak_lines[i].setData([], [])
        
        self.ui.label_10.setText(str(np.round(self.crystal.get_actual_value(), 3)))
        self.ui.label_10.setStyleSheet('color: green')

    def start_stop_scan(self):
        if self.ui.pb_start_scan.text() == "Stop":
            #self.timer_live.stop()
            if self.scanning is not None:
                #self.crystal.stop()
                self.scanning.kill = True
            self.plot_timer.stop()
            self.ui.pb_start_scan.setStyleSheet("color: rgb(255, 0, 0);")
            self.ui.pb_start_scan.setText("Start")
            self.scanning = None
        else:
            if self.ui.pb_start.text() == "Start":
                self.parent.error_box("Launch Spectrometer first")
                return
            #self.get_properties()
            if self.crystal is None:
                self.parent.error_box("Somethig wrong with crystal control server")
                return 
            self.image_scale()
            #str_range = str(self.ui.le_scan_range.text())
            #print("range = ", str_range)
            #try:
            #    val_range = eval(str_range)
            #    print(val_range)
            #except:
            #    self.parent.error_box("incorrect range")
            #    return


            self.scanning = ScanTool(mi=self.mi, crystal=self.crystal)
            self.scanning.cont_mode = True
            self.scanning.parent = self.parent
            self.scanning.timeout = self.ui.sbox_scan_wait.value()
            #self.scanning.val_range = val_range
            self.scanning.start()

            self.plot_timer.start(self.ui.sbox_scan_wait.value()*500)

            self.ui.pb_start_scan.setText("Stop")
            self.ui.pb_start_scan.setStyleSheet("color: rgb(85, 255, 127);")

    def show_hide_map(self):
        if self.ui.pb_show_map.text() == "Hide Map":
            self.ui.pb_show_map.setStyleSheet("color: rgb(85, 255, 255);")
            self.ui.pb_show_map.setText("Show Map")
            self.ui.widget_map.hide()
            self.ui.widget_scan.show()

        else:
            self.ui.pb_show_map.setText("Hide Map")
            self.ui.pb_show_map.setStyleSheet("color: rgb(85, 255, 127);")
            self.ui.widget_map.show()
            self.ui.widget_scan.hide()

    def add_image_widget(self):
        win = pg.GraphicsLayoutWidget()
        layout = QtGui.QGridLayout()
        self.ui.widget_map.setLayout(layout)
        layout.addWidget(win)

        self.img_plot = win.addPlot()

        self.add_image_item()

    def add_image_item(self):
        self.img_plot.clear()

        self.img_plot.setLabel('left', "", units='eV')
        self.img_plot.setLabel('bottom', "", units='deg')

        self.img = pg.ImageItem()

        self.img_plot.addItem(self.img)

        colormap = cm.get_cmap('viridis') #"nipy_spectral")  # cm.get_cmap("CMRmap")
        colormap._init()
        lut = (colormap._lut * 255).view(np.ndarray)  # Convert matplotlib colormap from 0-1 to 0 -255 for Qt

        # Apply the colormap
        self.img.setLookupTable(lut)


    def add_plot(self):
        win = pg.GraphicsLayoutWidget()
        # justify='right',,
        self.label = pg.LabelItem(justify='left', row=0, col=0)
        win.addItem(self.label)

        # self.plot1 = win.addPlot(row=0, col=0)
        self.plot1 = win.addPlot(row=1, col=0)

        self.label2 = pg.LabelItem(justify='right')
        win.addItem(self.label2, row=0, col=0)

        self.plot1.setLabel('left', "", units='eV')
        self.plot1.setLabel('bottom', "", units='deg')

        self.plot1.showGrid(1, 1, 1)

        self.plot1.getAxis('left').enableAutoSIPrefix(enable=False)  # stop the auto unit scaling on y axes
        layout = QtGui.QGridLayout()
        self.ui.widget_scan.setLayout(layout)
        layout.addWidget(win, 0, 0)

        self.plot1.setAutoVisible(y=True)

        self.plot1.addLegend()
        color = QtGui.QColor(0, 255, 255)
        pen = pg.mkPen(color, width=2)
        self.single = pg.PlotCurveItem(name='single')

        self.peak_lines = []
        self.peak_lines.append(pg.PlotCurveItem(pen=pg.mkPen((255, 51, 51), width=3), name='1-st peak', antialias=True))
        self.peak_lines.append(pg.PlotCurveItem(pen=pg.mkPen((51, 255, 51), width=3), name='2-st peak', antialias=True))
        self.peak_lines.append(pg.PlotCurveItem(pen=pg.mkPen((255, 255, 51), width=3), name='3-st peak', antialias=True))
        #self.peak_lines.append(pg.PlotCurveItem(pen=pg.mkPen((255, 255, 255), width=3), name='4-st peak', antialias=True))
        #self.peak_lines.append(pg.PlotCurveItem(pen=pg.mkPen((0, 255, 0), width=3), name='5-st peak', antialias=True))
        self.plot1.addItem(self.peak_lines[0])
        self.plot1.addItem(self.peak_lines[1])
        self.plot1.addItem(self.peak_lines[2])
        #self.plot1.addItem(self.peak_lines[3])
        #self.plot1.addItem(self.peak_lines[4])

        pen = pg.mkPen((0, 255, 255), width=2)

        self.fit_func = pg.PlotCurveItem(pen=pen, name='Gauss Fit')

        # self.plot1.addItem(self.fit_func)
        # self.plot1.enableAutoRange(False)
        # self.textItem = pg.TextItem(text="", border='w', fill=(0, 0, 0))
        # self.textItem.setPos(10, 10)

        # cross hair
        self.vLine = pg.InfiniteLine(angle=90, movable=False)
        self.hLine = pg.InfiniteLine(angle=0, movable=False)
        self.plot1.addItem(self.vLine, ignoreBounds=True)
        self.plot1.addItem(self.hLine, ignoreBounds=True)

    def mouseMoved(self, evt):
        #print("here", evt)
        #pos = evt.x(), evt.y() #evt[0]  ## using signal proxy turns original arguments into a tuple
        #print(evt.x())
        if self.plot1.sceneBoundingRect().contains(evt.x(), evt.y()):
            if self.scanning is not None:
                axis = self.scanning.doocs_vals
            else:
                return
            mousePoint = self.plot1.vb.mapSceneToView(evt)
            # index = int(mousePoint.x())
            array = np.asarray(axis)
            index = (np.abs(array - mousePoint.x())).argmin()
            #print(mousePoint.x(), index, len(self.x_axis))
            if index > 0 and index < len(axis):
                self.label.setText(
                    "<span style='font-size: 16pt', style='color: green'>x=%0.1f,   <span style='color: red'>y=%0.1f</span>" % (
                    mousePoint.x(), mousePoint.y()))
            self.vLine.setPos(mousePoint.x())
            self.hLine.setPos(mousePoint.y())