# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'UICalculator.ui'
#
# Created by: PyQt5 UI code generator 5.12
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_Form(object):
    def setupUi(self, Form):
        Form.setObjectName("Form")
        Form.resize(1296, 640)
        Form.setMinimumSize(QtCore.QSize(0, 0))
        Form.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        Form.setStyleSheet("background-color: white")
        self.gridLayout_2 = QtWidgets.QGridLayout(Form)
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.tabWidgetCalculator = QtWidgets.QTabWidget(Form)
        self.tabWidgetCalculator.setObjectName("tabWidgetCalculator")
        self.tab = QtWidgets.QWidget()
        self.tab.setObjectName("tab")
        self.gridLayout_7 = QtWidgets.QGridLayout(self.tab)
        self.gridLayout_7.setContentsMargins(10, 10, 10, 10)
        self.gridLayout_7.setSpacing(6)
        self.gridLayout_7.setObjectName("gridLayout_7")
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
        self.label = QtWidgets.QLabel(self.groupBox_5)
        self.label.setObjectName("label")
        self.gridLayout_10.addWidget(self.label, 0, 0, 1, 1)
        self.browse_button = QtWidgets.QPushButton(self.groupBox_5)
        self.browse_button.setObjectName("browse_button")
        self.gridLayout_10.addWidget(self.browse_button, 0, 4, 1, 1)
        self.file_name = QtWidgets.QLineEdit(self.groupBox_5)
        self.file_name.setObjectName("file_name")
        self.gridLayout_10.addWidget(self.file_name, 0, 1, 1, 3)
        self.roll_angle = QtWidgets.QDoubleSpinBox(self.groupBox_5)
        self.roll_angle.setObjectName("roll_angle")
        self.gridLayout_10.addWidget(self.roll_angle, 3, 1, 1, 2)
        self.label_3 = QtWidgets.QLabel(self.groupBox_5)
        self.label_3.setObjectName("label_3")
        self.gridLayout_10.addWidget(self.label_3, 3, 0, 1, 1)
        self.pb_start_calc = QtWidgets.QPushButton(self.groupBox_5)
        self.pb_start_calc.setMinimumSize(QtCore.QSize(200, 60))
        self.pb_start_calc.setMaximumSize(QtCore.QSize(1000, 1000))
        font = QtGui.QFont()
        font.setPointSize(14)
        font.setBold(True)
        font.setWeight(75)
        self.pb_start_calc.setFont(font)
        self.pb_start_calc.setStyleSheet("color: rgb(85, 255, 127);")
        self.pb_start_calc.setObjectName("pb_start_calc")
        self.gridLayout_10.addWidget(self.pb_start_calc, 4, 4, 1, 1)
        spacerItem = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.gridLayout_10.addItem(spacerItem, 4, 3, 1, 1)
        self.status = QtWidgets.QLabel(self.groupBox_5)
        self.status.setText("")
        self.status.setObjectName("status")
        self.gridLayout_10.addWidget(self.status, 4, 0, 1, 3)
        self.gridLayout_7.addWidget(self.groupBox_5, 2, 0, 2, 3)
        self.groupBox_2 = QtWidgets.QGroupBox(self.tab)
        self.groupBox_2.setObjectName("groupBox_2")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.groupBox_2)
        self.verticalLayout.setObjectName("verticalLayout")
        self.output = QtWidgets.QLabel(self.groupBox_2)
        self.output.setMinimumSize(QtCore.QSize(200, 20))
        font = QtGui.QFont()
        font.setBold(False)
        font.setWeight(50)
        self.output.setFont(font)
        self.output.setText("")
        self.output.setScaledContents(True)
        self.output.setWordWrap(True)
        self.output.setObjectName("output")
        self.verticalLayout.addWidget(self.output)
        self.gridLayout_7.addWidget(self.groupBox_2, 2, 3, 2, 3)
        self.widget_calc_2 = QtWidgets.QWidget(self.tab)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.MinimumExpanding, QtWidgets.QSizePolicy.MinimumExpanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.widget_calc_2.sizePolicy().hasHeightForWidth())
        self.widget_calc_2.setSizePolicy(sizePolicy)
        self.widget_calc_2.setMinimumSize(QtCore.QSize(300, 200))
        self.widget_calc_2.setObjectName("widget_calc_2")
        self.gridLayout_7.addWidget(self.widget_calc_2, 1, 3, 1, 3)
        self.widget_calc = QtWidgets.QWidget(self.tab)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.MinimumExpanding, QtWidgets.QSizePolicy.MinimumExpanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.widget_calc.sizePolicy().hasHeightForWidth())
        self.widget_calc.setSizePolicy(sizePolicy)
        self.widget_calc.setMinimumSize(QtCore.QSize(300, 200))
        self.widget_calc.setObjectName("widget_calc")
        self.gridLayout_7.addWidget(self.widget_calc, 1, 0, 1, 3)
        self.measured_title = QtWidgets.QTextEdit(self.tab)
        self.measured_title.setMaximumSize(QtCore.QSize(16777215, 40))
        self.measured_title.setObjectName("measured_title")
        self.gridLayout_7.addWidget(self.measured_title, 0, 0, 1, 1)
        self.model_title = QtWidgets.QTextEdit(self.tab)
        self.model_title.setMaximumSize(QtCore.QSize(608, 40))
        self.model_title.setObjectName("model_title")
        self.gridLayout_7.addWidget(self.model_title, 0, 3, 1, 1)
        self.tabWidgetCalculator.addTab(self.tab, "")
        self.gridLayout_2.addWidget(self.tabWidgetCalculator, 1, 0, 1, 1)

        self.retranslateUi(Form)
        self.tabWidgetCalculator.setCurrentIndex(0)
        QtCore.QMetaObject.connectSlotsByName(Form)

    def retranslateUi(self, Form):
        _translate = QtCore.QCoreApplication.translate
        Form.setWindowTitle(_translate("Form", "Self Seeding tools"))
        self.groupBox_5.setTitle(_translate("Form", "Calculate from file"))
        self.label.setText(_translate("Form", "File name:"))
        self.browse_button.setText(_translate("Form", "Browse"))
        self.label_3.setText(_translate("Form", "Roll angle:"))
        self.pb_start_calc.setText(_translate("Form", "Calculate from npz file"))
        self.groupBox_2.setTitle(_translate("Form", "Logs"))
        self.measured_title.setHtml(_translate("Form", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:\'.AppleSystemUIFont\'; font-size:13pt; font-weight:400; font-style:normal;\">\n"
"<p align=\"center\" style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-size:18pt; font-weight:600;\">Measurement</span></p></body></html>"))
        self.model_title.setHtml(_translate("Form", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:\'.AppleSystemUIFont\'; font-size:13pt; font-weight:400; font-style:normal;\">\n"
"<p align=\"center\" style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-size:18pt; font-weight:600;\">Model</span></p></body></html>"))
        self.tabWidgetCalculator.setTabText(self.tabWidgetCalculator.indexOf(self.tab), _translate("Form", "Calculator"))




if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    Form = QtWidgets.QWidget()
    ui = Ui_Form()
    ui.setupUi(Form)
    Form.show()
    sys.exit(app.exec_())
