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
        Form.resize(948, 613)
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
        self.widget_calc_2 = QtWidgets.QWidget(self.tab)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.MinimumExpanding, QtWidgets.QSizePolicy.MinimumExpanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.widget_calc_2.sizePolicy().hasHeightForWidth())
        self.widget_calc_2.setSizePolicy(sizePolicy)
        self.widget_calc_2.setMinimumSize(QtCore.QSize(300, 200))
        self.widget_calc_2.setObjectName("widget_calc_2")
        self.gridLayout_7.addWidget(self.widget_calc_2, 0, 1, 1, 1)
        self.widget_calc = QtWidgets.QWidget(self.tab)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.MinimumExpanding, QtWidgets.QSizePolicy.MinimumExpanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.widget_calc.sizePolicy().hasHeightForWidth())
        self.widget_calc.setSizePolicy(sizePolicy)
        self.widget_calc.setMinimumSize(QtCore.QSize(300, 200))
        self.widget_calc.setObjectName("widget_calc")
        self.gridLayout_7.addWidget(self.widget_calc, 0, 0, 1, 1)
        self.groupBox = QtWidgets.QGroupBox(self.tab)
        self.groupBox.setTitle("")
        self.groupBox.setObjectName("groupBox")
        self.gridLayout = QtWidgets.QGridLayout(self.groupBox)
        self.gridLayout.setObjectName("gridLayout")
        self.pb_start_calc = QtWidgets.QPushButton(self.groupBox)
        self.pb_start_calc.setMinimumSize(QtCore.QSize(200, 60))
        font = QtGui.QFont()
        font.setPointSize(18)
        font.setBold(True)
        font.setWeight(75)
        self.pb_start_calc.setFont(font)
        self.pb_start_calc.setStyleSheet("color: rgb(85, 255, 127);")
        self.pb_start_calc.setObjectName("pb_start_calc")
        self.gridLayout.addWidget(self.pb_start_calc, 1, 1, 1, 1)
        spacerItem = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.gridLayout.addItem(spacerItem, 0, 1, 1, 1)
        spacerItem1 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.gridLayout.addItem(spacerItem1, 2, 1, 1, 1)
        self.gridLayout_7.addWidget(self.groupBox, 3, 1, 1, 1)
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
        self.roll_angle = QtWidgets.QDoubleSpinBox(self.groupBox_5)
        self.roll_angle.setObjectName("roll_angle")
        self.gridLayout_10.addWidget(self.roll_angle, 2, 1, 1, 2)
        spacerItem2 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.gridLayout_10.addItem(spacerItem2, 3, 3, 1, 1)
        spacerItem3 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.gridLayout_10.addItem(spacerItem3, 2, 3, 1, 1)
        spacerItem4 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.gridLayout_10.addItem(spacerItem4, 2, 4, 1, 1)
        self.file_name = QtWidgets.QLineEdit(self.groupBox_5)
        self.file_name.setObjectName("file_name")
        self.gridLayout_10.addWidget(self.file_name, 0, 1, 1, 3)
        spacerItem5 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.gridLayout_10.addItem(spacerItem5, 3, 4, 1, 1)
        self.browse_button = QtWidgets.QPushButton(self.groupBox_5)
        self.browse_button.setObjectName("browse_button")
        self.gridLayout_10.addWidget(self.browse_button, 0, 4, 1, 1)
        self.label = QtWidgets.QLabel(self.groupBox_5)
        self.label.setObjectName("label")
        self.gridLayout_10.addWidget(self.label, 0, 0, 1, 1)
        self.label_3 = QtWidgets.QLabel(self.groupBox_5)
        self.label_3.setObjectName("label_3")
        self.gridLayout_10.addWidget(self.label_3, 2, 0, 1, 1)
        self.mono_no = QtWidgets.QLabel(self.groupBox_5)
        self.mono_no.setText("")
        self.mono_no.setObjectName("mono_no")
        self.gridLayout_10.addWidget(self.mono_no, 3, 0, 1, 3)
        self.gridLayout_7.addWidget(self.groupBox_5, 3, 0, 1, 1)
        self.tabWidgetCalculator.addTab(self.tab, "")
        self.gridLayout_2.addWidget(self.tabWidgetCalculator, 1, 0, 1, 1)

        self.retranslateUi(Form)
        self.tabWidgetCalculator.setCurrentIndex(0)
        QtCore.QMetaObject.connectSlotsByName(Form)

    def retranslateUi(self, Form):
        _translate = QtCore.QCoreApplication.translate
        Form.setWindowTitle(_translate("Form", "Energy Offset Calculator"))
        self.pb_start_calc.setText(_translate("Form", "Start"))
        self.groupBox_5.setTitle(_translate("Form", "Inputs"))
        self.browse_button.setText(_translate("Form", "Browse"))
        self.label.setText(_translate("Form", "File name:"))
        self.label_3.setText(_translate("Form", "Roll angle:"))
        self.tabWidgetCalculator.setTabText(self.tabWidgetCalculator.indexOf(self.tab), _translate("Form", "Calculator"))




if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    Form = QtWidgets.QWidget()
    ui = Ui_Form()
    ui.setupUi(Form)
    Form.show()
    sys.exit(app.exec_())