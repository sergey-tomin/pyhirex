"""
Most of GUI logic is placed here.
S.Tomin, 2017
"""

import json
import scipy
from PyQt5.QtGui import QPixmap, QImage, QScreen
from PyQt5 import QtWidgets
from PIL import Image
import subprocess
import base64
from datetime import datetime
import numpy as np
import sys
import os
import webbrowser
from shutil import copy
from PyQt5.QtWidgets import QCheckBox, QHBoxLayout, QMessageBox, QApplication, QMenu, QWidget, QAction, QTableWidget, QTableWidgetItem, QDoubleSpinBox

from gui.UICalculator import Ui_Form

from PyQt5 import QtGui, QtCore
from pathlib import Path
from opt_lib import hr_eV_s

from mint.xfel_interface import machine_readout_list
import time
try:
    _fromUtf8 = QtCore.QString.fromUtf8
except AttributeError:
    def _fromUtf8(s):
        return s

try:
    _encoding = QtGui.QApplication.UnicodeUTF8

    def _translate(context, text, disambig):
        return QtGui.QApplication.translate(context, text, disambig, _encoding)
except AttributeError:
    def _translate(context, text, disambig):
        return QtGui.QApplication.translate(context, text, disambig)


def send_to_desy_elog(author, title, severity, text, elog, image=None):
    """
    USED
    Send information to a supplied electronic logbook.
    Author: Christopher Behrens (DESY)
    """

    # The DOOCS elog expects an XML string in a particular format. This string
    # is beeing generated in the following as an initial list of strings.
    succeded = True  # indicator for a completely successful job
    # list beginning
    elogXMLStringList = [
        '<?xml version="1.0" encoding="ISO-8859-1"?>', '<entry>']

    # author information
    elogXMLStringList.append('<author>')
    elogXMLStringList.append(author)
    elogXMLStringList.append('</author>')
    # title information
    elogXMLStringList.append('<title>')
    elogXMLStringList.append(title)
    elogXMLStringList.append('</title>')
    # severity information
    elogXMLStringList.append('<severity>')
    elogXMLStringList.append(severity)
    elogXMLStringList.append('</severity>')
    # text information
    elogXMLStringList.append('<text>')
    elogXMLStringList.append(text)
    elogXMLStringList.append('</text>')
    # image information
    if image:
        try:
            #encodedImage = base64.b64encode(image)
            elogXMLStringList.append('<image>')
            elogXMLStringList.append(image)
            elogXMLStringList.append('</image>')
        except:  # make elog entry anyway, but return error (succeded = False)
            succeded = False
    # list end
    elogXMLStringList.append('</entry>')
    # join list to the final string
    elogXMLString = '\n'.join(elogXMLStringList)
    # open printer process
    try:
        lpr = subprocess.Popen(['/usr/bin/lp', '-o', 'raw', '-d', elog],
                               stdin=subprocess.PIPE, stdout=subprocess.PIPE)
        # send printer job
        lpr.communicate(elogXMLString.encode('utf-8'))
    except:
        succeded = False
    return succeded


class MainWindow(Ui_Form):
    def __init__(self, Form):
        Ui_Form.__init__(self)
        self.setupUi(Form)
        self.menubar.setNativeMenuBar(False)
        #self.mainToolBar.setVisible(False)
        self.Form = Form
        self.settings_file = self.Form.settings_file
        # load in the dark theme style sheet
        #if self.style_file != "standard.css":

        self.pb_calc_logbook.clicked.connect(lambda: self.log_cor2d(self.Form))

        style_index = self.get_style_name_index()
        style_name = self.Form.gui_styles[style_index]

        self.loadStyleSheet(filename=self.Form.gui_dir + style_name)

    def save_cor2d_file(self):
        Path(self.Form.data_dir).mkdir(parents=True, exist_ok=True)
        filename = self.Form.data_dir + \
            time.strftime("%Y%m%d-%H_%M_%S") + "_cor2d.npz"

        cor2d_tab = self.Form.corre2dtool
        np.savez(filename, dumpversion=1,
                 phen_scale=cor2d_tab.phen,
                 spec_hist=cor2d_tab.spec_hist,
                 doocs_vals_hist=cor2d_tab.doocs_vals_hist,
                 corr2d=cor2d_tab.spec_binned,
                 doocs_scale=cor2d_tab.doocs_bins,
                 doocs_channel=cor2d_tab.doocs_address_label,
                 )
        self.save_machine_status_file()

        return filename

    def save_machine_status_file(self):
    	import csv
    	filename = self.Form.data_dir + \
    	    time.strftime("%Y%m%d-%H_%M_%S") + "_status.txt"
    	with open(filename, 'w') as f:
            writer = csv.writer(f)
            writer.writerow(['address', 'value'])
            for value in machine_readout_list:
                writer.writerow([value, self.Form.mi.get_value(value)])
    	return filename

    def logbook(self, widget, text=""):
        """
        Method to send Optimization parameters + screenshot to eLogbook
        :return:
        """
        screenshot = self.get_screenshot(widget)
        device = self.Form.ui.combo_hirex.currentText()
        res = send_to_desy_elog(author="", title="pySpectrometer " + device, severity="INFO", text=text, elog=self.Form.mi.logbook_name,
                                image=screenshot)
        if not res:
            self.Form.error_box("error during eLogBook sending")

    def log_cor2d(self, widget):
        if self.Form.doocs_permit:
            filename = self.save_cor2d_file()
            #text = "Correlation2D data is saved in: " + filename
            #tmp
            filename_status = self.save_machine_status_file()
            text = "Correlation2D data is saved in: " + filename + \
                "\nMachine status data is saved in: " + filename_status
        else:
            text = ""
        self.logbook(widget, text=text)

    def get_screenshot(self, window_widget):
        screenshot_tmp = QtCore.QByteArray()
        screeshot_buffer = QtCore.QBuffer(screenshot_tmp)
        screeshot_buffer.open(QtCore.QIODevice.WriteOnly)
        widget = QtWidgets.QWidget.grab(window_widget)
        widget.save(screeshot_buffer, "png")
        return screenshot_tmp.toBase64().data().decode()

    def get_style_name_index(self):
        # pvs = self.ui.widget.pvs
        # check if file here
        if not os.path.isfile(self.settings_file):
            return 0

        with open(self.settings_file, 'r') as f:
            table = json.load(f)
        if "style" in table.keys():
            return table["style"]
        else:
            return 0

    def loadStyleSheet(self, filename="dark.css"):
        """
        Sets the dark GUI theme from a css file.
        :return:
        """
        try:
            self.cssfile = filename
            print("cssfile", self.cssfile)
            with open(self.cssfile, "r") as f:
                self.Form.setStyleSheet(f.read())
        except IOError:
            print('No style sheet found!')
