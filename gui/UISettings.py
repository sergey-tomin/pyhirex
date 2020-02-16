# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'UISettings.ui'
#
# Created by: PyQt5 UI code generator 5.9.2
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_Form(object):
    def setupUi(self, Form):
        Form.setObjectName("Form")
        Form.resize(619, 726)
        Form.setMinimumSize(QtCore.QSize(0, 0))
        Form.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        Form.setStyleSheet("background-color: white")
        self.gridLayout_2 = QtWidgets.QGridLayout(Form)
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        spacerItem = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout.addItem(spacerItem)
        self.pb_apply = QtWidgets.QPushButton(Form)
        self.pb_apply.setMinimumSize(QtCore.QSize(0, 0))
        font = QtGui.QFont()
        font.setPointSize(12)
        font.setBold(False)
        font.setWeight(50)
        self.pb_apply.setFont(font)
        self.pb_apply.setObjectName("pb_apply")
        self.horizontalLayout.addWidget(self.pb_apply)
        self.pb_cancel = QtWidgets.QPushButton(Form)
        self.pb_cancel.setMinimumSize(QtCore.QSize(0, 0))
        font = QtGui.QFont()
        font.setPointSize(12)
        font.setBold(False)
        font.setWeight(50)
        self.pb_cancel.setFont(font)
        self.pb_cancel.setObjectName("pb_cancel")
        self.horizontalLayout.addWidget(self.pb_cancel)
        self.gridLayout_2.addLayout(self.horizontalLayout, 1, 0, 1, 1)
        self.tabWidget = QtWidgets.QTabWidget(Form)
        self.tabWidget.setObjectName("tabWidget")
        self.tab = QtWidgets.QWidget()
        self.tab.setObjectName("tab")
        self.gridLayout_11 = QtWidgets.QGridLayout(self.tab)
        self.gridLayout_11.setObjectName("gridLayout_11")
        self.gridLayout = QtWidgets.QGridLayout()
        self.gridLayout.setObjectName("gridLayout")
        self.groupBox = QtWidgets.QGroupBox(self.tab)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Minimum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.groupBox.sizePolicy().hasHeightForWidth())
        self.groupBox.setSizePolicy(sizePolicy)
        self.groupBox.setMinimumSize(QtCore.QSize(0, 0))
        self.groupBox.setMaximumSize(QtCore.QSize(16777215, 16777215))
        self.groupBox.setObjectName("groupBox")
        self.gridLayout_4 = QtWidgets.QGridLayout(self.groupBox)
        self.gridLayout_4.setObjectName("gridLayout_4")
        self.label_3 = QtWidgets.QLabel(self.groupBox)
        self.label_3.setObjectName("label_3")
        self.gridLayout_4.addWidget(self.label_3, 5, 0, 1, 1)
        spacerItem1 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.gridLayout_4.addItem(spacerItem1, 7, 1, 1, 1)
        self.label = QtWidgets.QLabel(self.groupBox)
        self.label.setObjectName("label")
        self.gridLayout_4.addWidget(self.label, 7, 0, 1, 1)
        self.sb_hrx_npx = QtWidgets.QSpinBox(self.groupBox)
        self.sb_hrx_npx.setLocale(QtCore.QLocale(QtCore.QLocale.English, QtCore.QLocale.UnitedStates))
        self.sb_hrx_npx.setMinimum(10)
        self.sb_hrx_npx.setMaximum(5000)
        self.sb_hrx_npx.setProperty("value", 1280)
        self.sb_hrx_npx.setObjectName("sb_hrx_npx")
        self.gridLayout_4.addWidget(self.sb_hrx_npx, 7, 2, 1, 1)
        self.label_2 = QtWidgets.QLabel(self.groupBox)
        self.label_2.setObjectName("label_2")
        self.gridLayout_4.addWidget(self.label_2, 8, 0, 1, 1)
        self.le_ctrl_num_bunch = QtWidgets.QLineEdit(self.groupBox)
        font = QtGui.QFont()
        font.setPointSize(12)
        self.le_ctrl_num_bunch.setFont(font)
        self.le_ctrl_num_bunch.setLocale(QtCore.QLocale(QtCore.QLocale.English, QtCore.QLocale.UnitedStates))
        self.le_ctrl_num_bunch.setObjectName("le_ctrl_num_bunch")
        self.gridLayout_4.addWidget(self.le_ctrl_num_bunch, 8, 1, 1, 2)
        self.le_hirex_ch = QtWidgets.QLineEdit(self.groupBox)
        font = QtGui.QFont()
        font.setPointSize(12)
        self.le_hirex_ch.setFont(font)
        self.le_hirex_ch.setObjectName("le_hirex_ch")
        self.gridLayout_4.addWidget(self.le_hirex_ch, 2, 1, 1, 2)
        self.label_4 = QtWidgets.QLabel(self.groupBox)
        font = QtGui.QFont()
        font.setPointSize(12)
        self.label_4.setFont(font)
        self.label_4.setObjectName("label_4")
        self.gridLayout_4.addWidget(self.label_4, 2, 0, 1, 1)
        self.label_20 = QtWidgets.QLabel(self.groupBox)
        font = QtGui.QFont()
        font.setPointSize(12)
        self.label_20.setFont(font)
        self.label_20.setObjectName("label_20")
        self.gridLayout_4.addWidget(self.label_20, 4, 0, 1, 1)
        self.le_trans_ch = QtWidgets.QLineEdit(self.groupBox)
        font = QtGui.QFont()
        font.setPointSize(12)
        self.le_trans_ch.setFont(font)
        self.le_trans_ch.setText("")
        self.le_trans_ch.setObjectName("le_trans_ch")
        self.gridLayout_4.addWidget(self.le_trans_ch, 4, 1, 1, 2)
        self.label_6 = QtWidgets.QLabel(self.groupBox)
        self.label_6.setObjectName("label_6")
        self.gridLayout_4.addWidget(self.label_6, 6, 0, 1, 1)
        self.le_fast_xgm = QtWidgets.QLineEdit(self.groupBox)
        font = QtGui.QFont()
        font.setPointSize(12)
        self.le_fast_xgm.setFont(font)
        self.le_fast_xgm.setObjectName("le_fast_xgm")
        self.gridLayout_4.addWidget(self.le_fast_xgm, 5, 1, 1, 2)
        self.le_slow_xgm = QtWidgets.QLineEdit(self.groupBox)
        font = QtGui.QFont()
        font.setPointSize(12)
        self.le_slow_xgm.setFont(font)
        self.le_slow_xgm.setObjectName("le_slow_xgm")
        self.gridLayout_4.addWidget(self.le_slow_xgm, 6, 1, 1, 2)
        self.gridLayout.addWidget(self.groupBox, 3, 0, 1, 2)
        self.gridLayout_11.addLayout(self.gridLayout, 1, 0, 1, 1)
        self.groupBox_2 = QtWidgets.QGroupBox(self.tab)
        self.groupBox_2.setObjectName("groupBox_2")
        self.gridLayout_5 = QtWidgets.QGridLayout(self.groupBox_2)
        self.gridLayout_5.setObjectName("gridLayout_5")
        self.gridLayout_3 = QtWidgets.QGridLayout()
        self.gridLayout_3.setObjectName("gridLayout_3")
        self.label_12 = QtWidgets.QLabel(self.groupBox_2)
        self.label_12.setObjectName("label_12")
        self.gridLayout_3.addWidget(self.label_12, 5, 0, 1, 1)
        self.sb_hrx_npx_sa1 = QtWidgets.QSpinBox(self.groupBox_2)
        self.sb_hrx_npx_sa1.setMinimum(10)
        self.sb_hrx_npx_sa1.setMaximum(5000)
        self.sb_hrx_npx_sa1.setProperty("value", 1280)
        self.sb_hrx_npx_sa1.setObjectName("sb_hrx_npx_sa1")
        self.gridLayout_3.addWidget(self.sb_hrx_npx_sa1, 4, 2, 1, 1)
        self.label_9 = QtWidgets.QLabel(self.groupBox_2)
        self.label_9.setObjectName("label_9")
        self.gridLayout_3.addWidget(self.label_9, 2, 0, 1, 1)
        self.label_11 = QtWidgets.QLabel(self.groupBox_2)
        self.label_11.setObjectName("label_11")
        self.gridLayout_3.addWidget(self.label_11, 4, 0, 1, 1)
        self.label_8 = QtWidgets.QLabel(self.groupBox_2)
        self.label_8.setObjectName("label_8")
        self.gridLayout_3.addWidget(self.label_8, 1, 0, 1, 1)
        self.label_10 = QtWidgets.QLabel(self.groupBox_2)
        self.label_10.setObjectName("label_10")
        self.gridLayout_3.addWidget(self.label_10, 3, 0, 1, 1)
        self.label_7 = QtWidgets.QLabel(self.groupBox_2)
        self.label_7.setObjectName("label_7")
        self.gridLayout_3.addWidget(self.label_7, 0, 0, 1, 1)
        spacerItem2 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.gridLayout_3.addItem(spacerItem2, 4, 1, 1, 1)
        self.le_slow_xgm_sa1 = QtWidgets.QLineEdit(self.groupBox_2)
        self.le_slow_xgm_sa1.setObjectName("le_slow_xgm_sa1")
        self.gridLayout_3.addWidget(self.le_slow_xgm_sa1, 3, 1, 1, 2)
        self.le_fast_xgm_sa1 = QtWidgets.QLineEdit(self.groupBox_2)
        self.le_fast_xgm_sa1.setObjectName("le_fast_xgm_sa1")
        self.gridLayout_3.addWidget(self.le_fast_xgm_sa1, 2, 1, 1, 2)
        self.le_trans_ch_sa1 = QtWidgets.QLineEdit(self.groupBox_2)
        self.le_trans_ch_sa1.setObjectName("le_trans_ch_sa1")
        self.gridLayout_3.addWidget(self.le_trans_ch_sa1, 1, 1, 1, 2)
        self.le_hirex_ch_sa1 = QtWidgets.QLineEdit(self.groupBox_2)
        self.le_hirex_ch_sa1.setObjectName("le_hirex_ch_sa1")
        self.gridLayout_3.addWidget(self.le_hirex_ch_sa1, 0, 1, 1, 2)
        self.le_ctrl_num_bunch_sa1 = QtWidgets.QLineEdit(self.groupBox_2)
        self.le_ctrl_num_bunch_sa1.setObjectName("le_ctrl_num_bunch_sa1")
        self.gridLayout_3.addWidget(self.le_ctrl_num_bunch_sa1, 5, 1, 1, 2)
        self.gridLayout_5.addLayout(self.gridLayout_3, 0, 0, 1, 1)
        self.gridLayout_11.addWidget(self.groupBox_2, 2, 0, 1, 1)
        self.groupBox_3 = QtWidgets.QGroupBox(self.tab)
        self.groupBox_3.setObjectName("groupBox_3")
        self.gridLayout_7 = QtWidgets.QGridLayout(self.groupBox_3)
        self.gridLayout_7.setObjectName("gridLayout_7")
        self.gridLayout_6 = QtWidgets.QGridLayout()
        self.gridLayout_6.setObjectName("gridLayout_6")
        self.label_15 = QtWidgets.QLabel(self.groupBox_3)
        font = QtGui.QFont()
        font.setPointSize(12)
        self.label_15.setFont(font)
        self.label_15.setObjectName("label_15")
        self.gridLayout_6.addWidget(self.label_15, 1, 0, 1, 1)
        self.le_server = QtWidgets.QLineEdit(self.groupBox_3)
        font = QtGui.QFont()
        font.setPointSize(12)
        self.le_server.setFont(font)
        self.le_server.setObjectName("le_server")
        self.gridLayout_6.addWidget(self.le_server, 1, 2, 1, 1)
        self.combo_server = QtWidgets.QComboBox(self.groupBox_3)
        self.combo_server.setObjectName("combo_server")
        self.gridLayout_6.addWidget(self.combo_server, 1, 1, 1, 1)
        self.le_logbook = QtWidgets.QLineEdit(self.groupBox_3)
        font = QtGui.QFont()
        font.setPointSize(12)
        self.le_logbook.setFont(font)
        self.le_logbook.setObjectName("le_logbook")
        self.gridLayout_6.addWidget(self.le_logbook, 0, 2, 1, 1)
        spacerItem3 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.gridLayout_6.addItem(spacerItem3, 0, 1, 1, 1)
        self.label_5 = QtWidgets.QLabel(self.groupBox_3)
        font = QtGui.QFont()
        font.setPointSize(12)
        self.label_5.setFont(font)
        self.label_5.setObjectName("label_5")
        self.gridLayout_6.addWidget(self.label_5, 0, 0, 1, 1)
        self.gridLayout_7.addLayout(self.gridLayout_6, 0, 0, 1, 1)
        self.gridLayout_11.addWidget(self.groupBox_3, 0, 0, 1, 1)
        self.tabWidget.addTab(self.tab, "")
        self.tab_5 = QtWidgets.QWidget()
        self.tab_5.setObjectName("tab_5")
        self.gridLayout_10 = QtWidgets.QGridLayout(self.tab_5)
        self.gridLayout_10.setObjectName("gridLayout_10")
        self.label_18 = QtWidgets.QLabel(self.tab_5)
        self.label_18.setObjectName("label_18")
        self.gridLayout_10.addWidget(self.label_18, 0, 0, 1, 1)
        self.cb_style_def = QtWidgets.QComboBox(self.tab_5)
        self.cb_style_def.setObjectName("cb_style_def")
        self.gridLayout_10.addWidget(self.cb_style_def, 0, 1, 1, 1)
        spacerItem4 = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.gridLayout_10.addItem(spacerItem4, 1, 0, 1, 1)
        self.tabWidget.addTab(self.tab_5, "")
        self.gridLayout_2.addWidget(self.tabWidget, 0, 0, 1, 1)

        self.retranslateUi(Form)
        self.tabWidget.setCurrentIndex(0)
        QtCore.QMetaObject.connectSlotsByName(Form)

    def retranslateUi(self, Form):
        _translate = QtCore.QCoreApplication.translate
        Form.setWindowTitle(_translate("Form", "Ocelot Interface"))
        self.pb_apply.setText(_translate("Form", "Apply"))
        self.pb_cancel.setText(_translate("Form", "Cancel"))
        self.groupBox.setTitle(_translate("Form", "SASE2 HIREX"))
        self.label_3.setText(_translate("Form", "Fast XGM signal"))
        self.label.setText(_translate("Form", "HIREX N pixels"))
        self.label_2.setText(_translate("Form", "CTRL NUM Bunches"))
        self.le_ctrl_num_bunch.setText(_translate("Form", "XFEL.UTIL/BUNCH_PATTERN/CONTROL/NUM_BUNCHES_REQUESTED_2"))
        self.le_hirex_ch.setText(_translate("Form", "XFEL.EXP/DAQ.GOTTHARD_MASTER/SA2_XTD6_HIREX/ADC"))
        self.label_4.setText(_translate("Form", "HIREX signal"))
        self.label_20.setText(_translate("Form", "Transmission DOOCS "))
        self.label_6.setText(_translate("Form", "Slow XGM signal"))
        self.groupBox_2.setTitle(_translate("Form", "SASE1 HIREX"))
        self.label_12.setText(_translate("Form", "CTRL NUM Bunches"))
        self.label_9.setText(_translate("Form", "Fast XGM signal"))
        self.label_11.setText(_translate("Form", "HIREX N pixels"))
        self.label_8.setText(_translate("Form", "Transmission DOOCS "))
        self.label_10.setText(_translate("Form", "Slow XGM signal"))
        self.label_7.setText(_translate("Form", "HIREX sgnal"))
        self.groupBox_3.setTitle(_translate("Form", "GroupBox"))
        self.label_15.setText(_translate("Form", "Server"))
        self.le_server.setText(_translate("Form", "XFEL, XFEL_SIM"))
        self.le_logbook.setText(_translate("Form", "xfellog"))
        self.label_5.setText(_translate("Form", "LogBook"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab), _translate("Form", "General"))
        self.label_18.setText(_translate("Form", "Chose style by default"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_5), _translate("Form", "Appearance"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    Form = QtWidgets.QWidget()
    ui = Ui_Form()
    ui.setupUi(Form)
    Form.show()
    sys.exit(app.exec_())

