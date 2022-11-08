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
        self.settings_file = self.Form.settings_file
        # load in the dark theme style sheet
        #if self.style_file != "standard.css":

        #self.le_scan_doocs.editingFinished.connect(lambda : self.is_le_addr_ok(self.le_scan_doocs))
        self.le_doocs_ch_cor.editingFinished.connect(lambda: self.is_le_addr_ok(self.le_doocs_ch_cor))

        self.le_doocs_ch_hist.editingFinished.connect(lambda: self.is_le_addr_ok(self.le_doocs_ch_hist))
        #self.le_b.editingFinished.connect(self.check_address)
        self.pb_logbook.clicked.connect(lambda: self.log_waterflow(self.Form, save_data=1))
        # self.pb_logbook_only.clicked.connect(lambda: self.log_waterflow(self.Form, save_data=0))
        self.actionSend_to_logbook.triggered.connect(lambda: self.log_waterflow(self.Form, save_data=1))
        
        self.pb_logbook_cor2d.clicked.connect(lambda: self.log_cor2d(self.Form))
        self.actionSend_cor2d_to_logbook.triggered.connect(lambda: self.log_cor2d(self.Form))
        
        self.pb_logbook_specanalysis.clicked.connect(lambda: self.log_specanalysis(self.Form))
        self.actionSend_spec_analysis_to_logbook.triggered.connect(lambda: self.log_specanalysis(self.Form))
        
        style_index = self.get_style_name_index()
        style_name = self.Form.gui_styles[style_index]

        self.loadStyleSheet(filename=self.Form.gui_dir + style_name)

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
        table["sb_bnumber"] = self.sb_bnumber.value()
        table["sb_av_nbunch"] = self.sb_av_nbunch.value()
        table["sb_transmission"] = self.sb_transmission.value()
        table["sb_transmission_override"] = self.sb_transmission_override.isChecked()
        #table["le_b"] = str(self.le_b.text())

        table["sb_px1"] = self.sb_px1.value()
        table["sb_E0"] = self.sb_E0.value()
        table["sb_ev_px"] = self.sb_ev_px.value()
        #table["sb_nbunch_back"] = self.sb_nbunch_back.value()

        table["sbox_scan_wait"] = self.sbox_scan_wait.value()
        #table["le_scan_doocs"] = str(self.le_scan_doocs.text())
        table["le_scan_range"] = str(self.le_scan_range.text())
        table["le_doocs_ch_hist"] = str(self.le_doocs_ch_hist.text())
        table["le_doocs_ch_cor"] = str(self.le_doocs_ch_cor.text())
        # correlation
        table["le_doocs_ch_cor2d"] = self.le_doocs_ch_cor2d.text()
        table["sb_emin"] = self.sb_emin.value()
        table["sb_emax"] = self.sb_emax.value()
        if not self.Form.doocs_permit:
            print("Can not save State")
            return
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

        try:

            if "chb_a" in table.keys(): self.chb_a.setCheckState(table["chb_a"])
            if "chb_show_fit" in table.keys(): self.chb_show_fit.setCheckState(table["chb_show_fit"])
            if "sb_bnumber" in table.keys(): self.sb_bnumber.setValue(table["sb_bnumber"])
            if "sb_av_nbunch" in table.keys(): self.sb_av_nbunch.setValue(table["sb_av_nbunch"])
            if "sb_transmission" in table.keys(): self.sb_transmission.setValue(table["sb_transmission"])
            if "sb_transmission_override" in table.keys(): self.sb_transmission_override.setChecked(table["sb_transmission_override"])

            if "sb_px1" in table.keys(): self.sb_px1.setValue(table["sb_px1"])
            if "sb_E0" in table.keys(): self.sb_E0.setValue(table["sb_E0"])
            if "sb_ev_px" in table.keys(): self.sb_ev_px.setValue(table["sb_ev_px"])

            if "sbox_scan_wait" in table.keys(): self.sbox_scan_wait.setValue(table["sbox_scan_wait"])
            #if "le_scan_doocs" in table.keys(): self.le_scan_doocs.setText(table["le_scan_doocs"])
            if "le_scan_range" in table.keys(): self.le_scan_range.setText(table["le_scan_range"])
            if "le_doocs_ch_hist" in table.keys(): self.le_doocs_ch_hist.setText(table["le_doocs_ch_hist"])
            if "le_doocs_ch_cor" in table.keys(): self.le_doocs_ch_cor.setText(table["le_doocs_ch_cor"])
            # correlation
            if "le_doocs_ch_cor2d" in table.keys(): self.le_doocs_ch_cor2d.setText(table["le_doocs_ch_cor2d"])
            if "sb_emin" in table.keys(): self.sb_emin.setValue(table["sb_emin"])
            if "sb_emax" in table.keys(): self.sb_emax.setValue(table["sb_emax"])
            
            print("RESTORE STATE: OK")
        except:
            print("RESTORE STATE: ERROR")

    def save_cor2d_file(self):
        Path(self.Form.data_dir).mkdir(parents=True, exist_ok=True)
        filename = self.Form.data_dir + time.strftime("%Y%m%d-%H_%M_%S") + "_cor2d.npz"
        
        cor2d_tab = self.Form.corre2dtool
        np.savez(filename, dumpversion=1,
                phen_scale=cor2d_tab.phen, 
                spec_hist=cor2d_tab.spec_hist, 
                doocs_vals_hist=cor2d_tab.doocs_vals_hist, 
                corr2d=cor2d_tab.spec_binned, 
                doocs_scale = cor2d_tab.doocs_bins, 
                doocs_channel=cor2d_tab.doocs_address_label,
                )
                
        #self.save_machine_status_file(filename = filename+"_status.txt")
        
        return filename
        
    def save_machine_status_file(self, filename=None):
    	import csv
    	if filename == None:
    		filename = self.Form.data_dir + time.strftime("%Y%m%d-%H_%M_%S") + "_status.txt"
    	with open(filename,'w') as f:
    		writer = csv.writer(f)
    		writer.writerow(['address','value'])
    		for value in machine_readout_list:
    			writer.writerow([value,self.Form.mi.get_value(value)])
    	return filename

    def save_waterflow_file(self):
        Path(self.Form.data_dir).mkdir(parents=True, exist_ok=True)
        filename = self.Form.data_dir + time.strftime("%Y%m%d-%H_%M_%S") + "_waterflow.npz"
        np.savez(filename, dumpversion=2,
        e_axis_uncut=self.Form.x_axis,
        e_axis=self.Form.x_axis_disp,
        cut_px_first=self.Form.px_first, 
        cut_px_last=self.Form.px_last,
        average=self.Form.ave_spectrum, 
        map=self.Form.data_2d,
        background=self.Form.background)
        return filename
        
    def save_specanalysis_file(self):
        Path(self.Form.data_dir).mkdir(parents=True, exist_ok=True)
        specanalysis_tab = self.Form.analysistool
        filename = self.Form.data_dir + time.strftime("%Y%m%d-%H_%M_%S") + "_specanalysis.npz"
        print('save specanalysis to file ', filename)

        np.savez(filename, dumpversion=1,
        phen_scale=specanalysis_tab.spar.phen,
        spec_hist=specanalysis_tab.spar.spec,
        calib_energy_coef=self.Form.calib_energy_coef,
        corrn_center=specanalysis_tab.corrn.corr,
        corrn_center_dphen=specanalysis_tab.corrn.domega * hr_eV_s,
        corrn_center_phen=specanalysis_tab.corrn.omega * hr_eV_s,
        fit_t=specanalysis_tab.g2fit.fit_t,
        fit_t_comp=specanalysis_tab.g2fit.fit_t_comp,
        fit_t_contr=specanalysis_tab.g2fit.fit_contrast,
        fit_t_pedestal=specanalysis_tab.g2fit.fit_pedestal)

        return filename

    def logbook(self, widget, text=""):
        """
        Method to send Optimization parameters + screenshot to eLogbook
        :return:
        """
        screenshot = self.get_screenshot(widget)
        device = self.Form.ui.combo_hirex.currentText()
        res = send_to_desy_elog(author="", title="pySpectrometer "+ device, severity="INFO", text=text, elog=self.Form.mi.logbook_name,
                          image=screenshot)
        if not res:
            self.Form.error_box("error during eLogBook sending")

    def log_waterflow(self, widget, save_data=1):
        if self.Form.doocs_permit and save_data:
            filename = self.save_waterflow_file()
            text = "Waterfall data is saved in: " + filename
            
        else:
            text = ""
        self.logbook(widget, text=text)

    def log_cor2d(self, widget):
        if self.Form.doocs_permit:
            filename = self.save_cor2d_file()
            #text = "Correlation2D data is saved in: " + filename
            #tmp
            filename_status = self.save_machine_status_file(filename = filename+"_status.txt")
            text = "Correlation2D data is saved in: " + filename + "\nMachine status data is saved in: " + filename_status
        else:
            text = ""
        self.logbook(widget, text=text)
        
    def log_specanalysis(self, widget):
        self.save_specanalysis_file()
        if self.Form.doocs_permit:
            filename = self.save_specanalysis_file()
            text = "Spectrum analysis: " + filename
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
            print("cssfile" , self.cssfile)
            with open(self.cssfile, "r") as f:
                self.Form.setStyleSheet(f.read())
        except IOError:
            print ('No style sheet found!')