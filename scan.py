import numpy as np
import pyqtgraph as pg
from threading import Thread, Event
from PyQt5 import QtGui, QtCore
import time
from mint.opt_objects import Device



class ScanTool(Thread):
    def __init__(self, mi, device):
        super(ScanTool, self).__init__()
        self.mi = mi
        self.devmode = False
        self.device = device
        self.timeout = 2
        self.val_range = []
        self.kill = False
        self._stop_event = Event()

    def load(self):
        self.background = np.array([])
        try:
            self.background = np.loadtxt("background.txt")
        except Exception as ex:
            print("Problem with background: {}. Exception was: {}".format("background.txt", ex))

        return self.background

    def run(self):
        print("run = ", self.val_range)
        for val in self.val_range:
            if self.kill:
                return
            print(self.device.id, "<--", val)
            #x = self.device.set_value(val)
            time.sleep(self.timeout)

        print("Background finished")

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
        # self.plot1.scene().sigMouseMoved.connect(self.mouseMoved)

        self.scanning = None
        self.ui.pb_start_scan.clicked.connect(self.start_stop_scan)
        self.ui.pb_check_range.clicked.connect(self.check_range)


    def check_range(self):
        str_range = str(self.ui.le_scan_range.text())
        try:
            x = eval(str_range)
            self.parent.error_box(str(x))
        except:
            self.parent.error_box("incorrect range")
            return



    def start_stop_scan(self):
        if self.ui.pb_start_scan.text() == "Stop":
            #self.timer_live.stop()
            if self.scanning is not None:
                self.scanning.kill = True

            self.ui.pb_start_scan.setStyleSheet("color: rgb(255, 0, 0);")
            self.ui.pb_start_scan.setText("Start")
            self.scanning = None
        else:
            doocs_channel = str(self.ui.le_scan_doocs.text())
            print("DOOCS", doocs_channel)
            str_range = str(self.ui.le_scan_range.text())
            print("range = ", str_range)
            try:
                val_range = eval(str_range)
                print(val_range)
            except:
                self.parent.error_box("incorrect range")
                return


            self.scanning = ScanTool(mi=self.mi, device=Device(eid=doocs_channel))
            self.scanning.timeout = self.ui.sbox_scan_wait.value()
            self.scanning.val_range = val_range
            self.scanning.start()
            self.ui.pb_start_scan.setText("Stop")
            self.ui.pb_start_scan.setStyleSheet("color: rgb(85, 255, 127);")


    def add_plot(self):
        win = pg.GraphicsLayoutWidget()
        # justify='right',,
        self.label = pg.LabelItem(justify='left', row=0, col=0)
        win.addItem(self.label)

        # self.plot1 = win.addPlot(row=0, col=0)
        self.plot1 = win.addPlot(row=1, col=0)

        self.label2 = pg.LabelItem(justify='right')
        win.addItem(self.label2, row=0, col=0)

        self.plot1.setLabel('left', "A", units='au')
        self.plot1.setLabel('bottom', "", units='eV')

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

        self.plot1.addItem(self.single)

        color = QtGui.QColor(255, 0, 0)
        pen = pg.mkPen((255, 0, 0), width=2)
        # self.average = pg.PlotCurveItem(x=[], y=[], pen=pen, name='average')
        self.average = pg.PlotCurveItem(pen=pen, name='average')

        self.plot1.addItem(self.average)

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