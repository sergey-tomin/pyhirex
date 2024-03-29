# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'UILogger.ui'
#
# Created by: PyQt5 UI code generator 5.9.2
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_Form(object):
    def setupUi(self, Form):
        Form.setObjectName("Form")
        Form.resize(1223, 861)
        Form.setMinimumSize(QtCore.QSize(0, 0))
        Form.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        Form.setStyleSheet("background-color: white")
        self.gridLayout_2 = QtWidgets.QGridLayout(Form)
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.tabWidgetLogger = QtWidgets.QTabWidget(Form)
        self.tabWidgetLogger.setObjectName("tabWidgetLogger")
        self.tab = QtWidgets.QWidget()
        self.tab.setObjectName("tab")
        self.gridLayout_7 = QtWidgets.QGridLayout(self.tab)
        self.gridLayout_7.setContentsMargins(10, 10, 10, 10)
        self.gridLayout_7.setSpacing(6)
        self.gridLayout_7.setObjectName("gridLayout_7")
        self.widget_log = QtWidgets.QWidget(self.tab)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.MinimumExpanding, QtWidgets.QSizePolicy.MinimumExpanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.widget_log.sizePolicy().hasHeightForWidth())
        self.widget_log.setSizePolicy(sizePolicy)
        self.widget_log.setMinimumSize(QtCore.QSize(300, 200))
        self.widget_log.setObjectName("widget_log")
        self.gridLayout_7.addWidget(self.widget_log, 0, 0, 1, 2)
        self.groupBox_5 = QtWidgets.QGroupBox(self.tab)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Minimum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.groupBox_5.sizePolicy().hasHeightForWidth())
        self.groupBox_5.setSizePolicy(sizePolicy)
        self.groupBox_5.setObjectName("groupBox_5")
        self.gridLayout_10 = QtWidgets.QGridLayout(self.groupBox_5)
        self.gridLayout_10.setContentsMargins(5, 5, 5, 5)
        self.gridLayout_10.setSpacing(6)
        self.gridLayout_10.setObjectName("gridLayout_10")
        self.le_b = QtWidgets.QLineEdit(self.groupBox_5)
        self.le_b.setEnabled(False)
        self.le_b.setMinimumSize(QtCore.QSize(0, 25))
        self.le_b.setObjectName("le_b")
        self.gridLayout_10.addWidget(self.le_b, 3, 0, 1, 1)
        self.combo_log_ch_a = QtWidgets.QComboBox(self.groupBox_5)
        self.combo_log_ch_a.setMinimumSize(QtCore.QSize(0, 25))
        self.combo_log_ch_a.setObjectName("combo_log_ch_a")
        self.gridLayout_10.addWidget(self.combo_log_ch_a, 0, 0, 1, 1)
        self.le_a = QtWidgets.QLineEdit(self.groupBox_5)
        self.le_a.setEnabled(False)
        self.le_a.setMinimumSize(QtCore.QSize(0, 25))
        self.le_a.setObjectName("le_a")
        self.gridLayout_10.addWidget(self.le_a, 2, 0, 1, 1)
        self.combo_log_ch_b = QtWidgets.QComboBox(self.groupBox_5)
        self.combo_log_ch_b.setMinimumSize(QtCore.QSize(0, 25))
        self.combo_log_ch_b.setObjectName("combo_log_ch_b")
        self.gridLayout_10.addWidget(self.combo_log_ch_b, 1, 0, 1, 1)
        self.gridLayout_7.addWidget(self.groupBox_5, 3, 0, 1, 1)
        self.groupBox = QtWidgets.QGroupBox(self.tab)
        self.groupBox.setObjectName("groupBox")
        self.gridLayout = QtWidgets.QGridLayout(self.groupBox)
        self.gridLayout.setObjectName("gridLayout")
        self.sb_nponts = QtWidgets.QSpinBox(self.groupBox)
        self.sb_nponts.setMinimum(100)
        self.sb_nponts.setMaximum(10000)
        self.sb_nponts.setSingleStep(10)
        self.sb_nponts.setProperty("value", 1000)
        self.sb_nponts.setObjectName("sb_nponts")
        self.gridLayout.addWidget(self.sb_nponts, 0, 1, 1, 1)
        self.label = QtWidgets.QLabel(self.groupBox)
        self.label.setObjectName("label")
        self.gridLayout.addWidget(self.label, 0, 0, 1, 1)
        self.pb_start_log = QtWidgets.QPushButton(self.groupBox)
        self.pb_start_log.setMinimumSize(QtCore.QSize(200, 60))
        font = QtGui.QFont()
        font.setPointSize(18)
        font.setBold(True)
        font.setWeight(75)
        self.pb_start_log.setFont(font)
        self.pb_start_log.setStyleSheet("color: rgb(85, 255, 127);")
        self.pb_start_log.setObjectName("pb_start_log")
        self.gridLayout.addWidget(self.pb_start_log, 1, 0, 1, 2)
        self.gridLayout_7.addWidget(self.groupBox, 3, 1, 1, 1)
        self.tabWidgetLogger.addTab(self.tab, "")
        self.gridLayout_2.addWidget(self.tabWidgetLogger, 1, 0, 1, 1)

        self.retranslateUi(Form)
        self.tabWidgetLogger.setCurrentIndex(0)
        QtCore.QMetaObject.connectSlotsByName(Form)

    def retranslateUi(self, Form):
        _translate = QtCore.QCoreApplication.translate
        Form.setWindowTitle(_translate("Form", "Ocelot Interface"))
        self.groupBox_5.setTitle(_translate("Form", "Channels"))
        self.groupBox.setTitle(_translate("Form", "Control"))
        self.label.setText(_translate("Form", "Npoints"))
        self.pb_start_log.setText(_translate("Form", "Start"))
        self.tabWidgetLogger.setTabText(self.tabWidgetLogger.indexOf(self.tab), _translate("Form", "Logger"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    Form = QtWidgets.QWidget()
    ui = Ui_Form()
    ui.setupUi(Form)
    Form.show()
    sys.exit(app.exec_())

