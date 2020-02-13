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
from PyQt5.QtWidgets import QCheckBox, QHBoxLayout, QMessageBox, QApplication,QMenu, QWidget, QAction, QTableWidget, QTableWidgetItem, QDoubleSpinBox

from gui.UISpectrometer import Ui_MainWindow

from PyQt5 import QtGui, QtCore

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
    Send information to a supplied electronic logbook.
    Author: Christopher Behrens (DESY)
    """

    # The DOOCS elog expects an XML string in a particular format. This string
    # is beeing generated in the following as an initial list of strings.
    succeded = True  # indicator for a completely successful job
    # list beginning
    elogXMLStringList = ['<?xml version="1.0" encoding="ISO-8859-1"?>', '<entry>']

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



class MainWindow(Ui_MainWindow):
    def __init__(self, Form):
        Ui_MainWindow.__init__(self)
        self.setupUi(Form)
        self.menubar.setNativeMenuBar(False)
        #self.mainToolBar.setVisible(False)
        self.Form = Form
        # load in the dark theme style sheet
        #if self.style_file != "standard.css":
        self.le_a.editingFinished.connect(self.check_address)
        self.le_scan_doocs.editingFinished.connect(lambda : self.is_le_addr_ok(self.le_scan_doocs))
        #self.le_b.editingFinished.connect(self.check_address)
        self.pb_logbook.clicked.connect(lambda: self.logbook(self.Form))
        self.loadStyleSheet(filename=self.Form.gui_dir +"colinDark.css")

    def check_address(self):
        self.is_le_addr_ok(self.le_a)
        #self.is_le_addr_ok(self.le_b)


    def is_le_addr_ok(self, line_edit):
        if not line_edit.isEnabled():
            return False
        dev = str(line_edit.text())
        state = True
        try:
            val = self.Form.mi.get_value(dev)
            if val is None:
                state = False
        except:
            state = False

        if state:
            line_edit.setStyleSheet("color: rgb(85, 255, 0);")
        else:
            line_edit.setStyleSheet("color: red")
        line_edit.clearFocus()
        return state

    def save_state(self, filename):
        # pvs = self.ui.widget.pvs
        #table = self.widget.get_state()
        table = {}
        table["chb_a"] = self.chb_a.checkState()
        table["chb_show_fit"] = self.chb_show_fit.checkState()
        table["comboBox"] = self.comboBox.currentIndex()
        table["sb_bnumber"] = self.sb_bnumber.value()
        table["sb_av_nbunch"] = self.sb_av_nbunch.value()
        table["sb_transmission"] = self.sb_transmission.value()
        table["le_a"] = str(self.le_a.text())
        #table["le_b"] = str(self.le_b.text())

        table["sb_px1"] = self.sb_px1.value()
        table["sb_E0"] = self.sb_E0.value()
        table["sb_ev_px"] = self.sb_ev_px.value()
        table["sb_nbunch_back"] = self.sb_nbunch_back.value()


        with open(filename, 'w') as f:
            json.dump(table, f)
        # pickle.dump(table, filename)
        print("SAVE State")

    def restore_state(self, filename):
        try:
            with open(filename, 'r') as f:
                # data_new = pickle.load(f)
                table = json.load(f)
        except Exception as ex:
            print("Restore State failed for file: {}. Exception was: {}".format(filename, ex))
            return


        # Build the PV list from dev PVs or selected source
        #pvs = table["id"]
        #self.widget.set_machine_interface(self.Form.mi)
        #self.widget.getPvList(pvs)
        ## set checkbot status
        #self.widget.uncheckBoxes()
        #self.widget.set_state(table)

        try:

            if "chb_a" in table.keys(): self.chb_a.setCheckState(table["chb_a"])
            if "chb_show_fit" in table.keys(): self.chb_show_fit.setCheckState(table["chb_show_fit"])
            if "comboBox" in table.keys(): self.comboBox.setCurrentIndex(table["comboBox"])
            if "sb_bnumber" in table.keys(): self.sb_bnumber.setValue(table["sb_bnumber"])
            if "sb_av_nbunch" in table.keys(): self.sb_av_nbunch.setValue(table["sb_av_nbunch"])
            if "sb_transmission" in table.keys(): self.sb_transmission.setValue(table["sb_transmission"])

            if "sb_px1" in table.keys(): self.sb_px1.setValue(table["sb_px1"])
            if "sb_E0" in table.keys(): self.sb_E0.setValue(table["sb_E0"])
            if "sb_ev_px" in table.keys(): self.sb_ev_px.setValue(table["sb_ev_px"])
            if "sb_nbunch_back" in table.keys(): self.sb_nbunch_back.setValue(table["sb_nbunch_back"])
            
            
            if "le_a" in table.keys(): self.le_a.setText(table["le_a"])
            #if "le_b" in table.keys(): self.le_b.setText(table["le_b"])

            print("RESTORE STATE: OK")
        except:
            print("RESTORE STATE: ERROR")

    def logbook(self, widget):
        """
        Method to send Optimization parameters + screenshot to eLogboob
        :return:
        """

        filename = "screenshot"
        filetype = "png"
        #self.screenShot(filename, filetype)

        # curr_time = datetime.now()
        # timeString = curr_time.strftime("%Y-%m-%dT%H:%M:%S")
        text = ""


        #screenshot = open(self.Form.optimizer_path + filename + "." + filetype, 'rb')
        
        screenshot = self.get_screenshot(widget)
        #res = send_to_desy_elog(author="", title="OCELOT Correction tool", severity="INFO", text=text, elog=self.Form.logbook,
        #                  image=screenshot.read())
        
        res = send_to_desy_elog(author="", title="pySpectrometer", severity="INFO", text=text, elog=self.Form.mi.logbook_name,
                          image=screenshot)

        if not res:
            self.Form.error_box("error during eLogBook sending")

    def get_screenshot(self, window_widget):
        screenshot_tmp = QtCore.QByteArray()
        screeshot_buffer = QtCore.QBuffer(screenshot_tmp)
        screeshot_buffer.open(QtCore.QIODevice.WriteOnly)
        widget = QtWidgets.QWidget.grab(window_widget)
        widget.save(screeshot_buffer, "png")
        return screenshot_tmp.toBase64().data().decode()

    def loadStyleSheet(self, filename="dark.css"):
        """
        Sets the dark GUI theme from a css file.
        :return:
        """
        try:

            self.cssfile = filename
            with open(self.cssfile, "r") as f:
                self.Form.setStyleSheet(f.read())
        except IOError:
            print ('No style sheet found!')