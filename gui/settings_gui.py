"""
Settings. S.Tomin.
"""

import sys
from PyQt5.QtWidgets import QCheckBox, QHBoxLayout, QMessageBox, QApplication,QMenu, QWidget, QAction, QTableWidget, QTableWidgetItem, QDoubleSpinBox
import os
from PyQt5.QtGui import QIcon
from PyQt5.QtCore import pyqtSlot
from gui.UISettings import Ui_Form
import json
#import logging
#logger = logging.getLogger(__name__)

class HirexSettings(QWidget):
    def __init__(self, parent=None):
        super().__init__()
        #QWidget.__init__(self, parent)
        self.master = parent
        self.ui = Ui_Form()
        self.ui.setupUi(self)

        self.ui.pb_apply.clicked.connect(self.apply_settings)

        self.ui.pb_cancel.clicked.connect(self.close)
        self.ui.cb_style_def.addItems(self.master.gui_styles)
        #self.ui.cb_style_def.addItem("standard.css")
        #self.ui.cb_style_def.addItem("colinDark.css")
        #self.ui.cb_style_def.addItem("dark.css")
        self.style_file = None
        self.load_state(filename=self.master.settings_file)
        #except:
        #    pass
        if self.style_file is not None:
            self.loadStyleSheet(filename=self.style_file)

    def apply_settings(self):
        update = self.question_box("Save Settings and Close?")
        if update:
            if not os.path.exists(self.master.settings_file):
                os.makedirs(self.master.settings_file)
            self.save_state(self.master.settings_file)
            self.master.load_settings()
            self.close()

    def save_state(self, filename):
        table = {}

        table["logbook"] = self.ui.le_logbook.text()

        table["le_trans_ch"] = self.ui.le_trans_ch.text()
        table["le_hirex_ch"] = self.ui.le_hirex_ch.text()
        table["sb_hrx_npx"] = self.ui.sb_hrx_npx.value()
        table["le_ctrl_num_bunch"] = self.ui.le_ctrl_num_bunch.text()

        table["server_list"] = self.string2list(self.ui.le_server.text())
        table["server"] = self.ui.combo_server.currentText()

        table["le_fast_xgm"] = self.ui.le_fast_xgm.text()
        table["le_slow_xgm"] = self.ui.le_slow_xgm.text()

        table["style"] = self.ui.cb_style_def.currentIndex()
        table["style_file"] = self.ui.cb_style_def.currentText()

        table["le_trans_ch_sa1"] = self.ui.le_trans_ch_sa1.text()
        table["le_hirex_ch_sa1"] = self.ui.le_hirex_ch_sa1.text()
        table["sb_hrx_npx_sa1"] = self.ui.sb_hrx_npx_sa1.value()
        table["le_ctrl_num_bunch_sa1"] = self.ui.le_ctrl_num_bunch_sa1.text()
        table["le_fast_xgm_sa1"] = self.ui.le_fast_xgm_sa1.text()
        table["le_slow_xgm_sa1"] = self.ui.le_slow_xgm_sa1.text()
        
        table["le_2d_hist_size"] = int(self.ui.le_2d_hist_size.text())

        with open(filename, 'w') as f:
            json.dump(table, f)
        print("SAVE State")

    def list2string(self, dev_list):
        return ", ".join(dev_list)
    
    def string2list(self, dev_str):
        lst = dev_str.split(",")
        lst = [text.replace(" ", "") for text in lst]
        lst = [text.replace("\n", "") for text in lst]
        return lst

    def load_state(self, filename):
        # pvs = self.ui.widget.pvs
        # check if file here
        if not os.path.isfile(filename):
            return

        with open(filename, 'r') as f:
            table = json.load(f)

        if "le_hirex_ch" in table.keys(): self.ui.le_hirex_ch.setText(table["le_hirex_ch"])
        if "le_trans_ch" in table.keys(): self.ui.le_trans_ch.setText(table["le_trans_ch"])
        if "sb_hrx_npx" in table.keys():  self.ui.sb_hrx_npx.setValue(table["sb_hrx_npx"])
        if "le_ctrl_num_bunch" in table.keys(): self.ui.le_ctrl_num_bunch.setText(table["le_ctrl_num_bunch"])

        if "le_logbook" in table.keys(): self.ui.le_logbook.setText(table["le_logbook"])

        if "le_fast_xgm" in table.keys(): self.ui.le_fast_xgm.setText(table["le_fast_xgm"])
        if "le_slow_xgm" in table.keys(): self.ui.le_slow_xgm.setText(table["le_slow_xgm"])



        if "le_hirex_ch_sa1" in table.keys(): self.ui.le_hirex_ch_sa1.setText(table["le_hirex_ch_sa1"])
        if "le_trans_ch_sa1" in table.keys(): self.ui.le_trans_ch_sa1.setText(table["le_trans_ch_sa1"])
        if "sb_hrx_npx_sa1" in table.keys():  self.ui.sb_hrx_npx_sa1.setValue(table["sb_hrx_npx_sa1"])
        if "le_ctrl_num_bunch_sa1" in table.keys(): self.ui.le_ctrl_num_bunch_sa1.setText(table["le_ctrl_num_bunch_sa1"])
        if "le_fast_xgm_sa1" in table.keys(): self.ui.le_fast_xgm_sa1.setText(table["le_fast_xgm_sa1"])
        if "le_slow_xgm_sa1" in table.keys(): self.ui.le_slow_xgm_sa1.setText(table["le_slow_xgm_sa1"])
        
        if "le_2d_hist_size" in table.keys(): self.ui.le_2d_hist_size.setText(str(table["le_2d_hist_size"]))

        if "server_list" in table.keys():
            self.ui.le_server.setText(self.list2string(table["server_list"]))
            for name in table["server_list"]:
                self.ui.combo_server.addItem(name)
            if "server" in table.keys() and table["server"] in table["server_list"]:
                indx = table["server_list"].index(table["server"])
            else:
                indx = 0
            self.ui.combo_server.setCurrentIndex(indx)

        if "style" in table.keys(): self.ui.cb_style_def.setCurrentIndex(table["style"])
        self.style_file = self.ui.cb_style_def.currentText()

        print("LOAD State")


    def question_box(self, message):
        #QtGui.QMessageBox.question(self, "Question box", message)
        reply = QMessageBox.question(self, "Question Box",
                message,
                QMessageBox.Yes | QMessageBox.No)
        if reply == QMessageBox.Yes:
            return True

        return False

    def loadStyleSheet(self, filename):
        """ Load in the dark theme style sheet. """
        try:
            self.cssfile = self.master.gui_dir + filename
            print(self.cssfile)
            with open(self.cssfile, "r") as f:
                #print(f)
                self.setStyleSheet(f.read())
        except IOError:
            print('No style sheet found!')


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = HirexSettings()
    window.show()
    sys.exit(app.exec_())