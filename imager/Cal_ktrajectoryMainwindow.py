# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'Cal_ktrajectoryMainwindow.ui'
#
# Created by: PyQt5 UI code generator 5.6
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(942, 737)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.listWidget_G_read = QtWidgets.QListWidget(self.centralwidget)
        self.listWidget_G_read.setGeometry(QtCore.QRect(60, 110, 31, 61))
        self.listWidget_G_read.setSelectionRectVisible(True)
        self.listWidget_G_read.setObjectName("listWidget_G_read")
        item = QtWidgets.QListWidgetItem()
        self.listWidget_G_read.addItem(item)
        item = QtWidgets.QListWidgetItem()
        self.listWidget_G_read.addItem(item)
        item = QtWidgets.QListWidgetItem()
        self.listWidget_G_read.addItem(item)
        self.pushButton_CalKtra_GIRF = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_CalKtra_GIRF.setGeometry(QtCore.QRect(550, 490, 211, 71))
        self.pushButton_CalKtra_GIRF.setObjectName("pushButton_CalKtra_GIRF")
        self.listWidget_G_encode1 = QtWidgets.QListWidget(self.centralwidget)
        self.listWidget_G_encode1.setGeometry(QtCore.QRect(150, 110, 31, 61))
        self.listWidget_G_encode1.setSelectionRectVisible(True)
        self.listWidget_G_encode1.setObjectName("listWidget_G_encode1")
        item = QtWidgets.QListWidgetItem()
        self.listWidget_G_encode1.addItem(item)
        item = QtWidgets.QListWidgetItem()
        self.listWidget_G_encode1.addItem(item)
        item = QtWidgets.QListWidgetItem()
        self.listWidget_G_encode1.addItem(item)
        self.gridLayoutWidget = QtWidgets.QWidget(self.centralwidget)
        self.gridLayoutWidget.setGeometry(QtCore.QRect(70, 320, 861, 71))
        self.gridLayoutWidget.setObjectName("gridLayoutWidget")
        self.gridLayout = QtWidgets.QGridLayout(self.gridLayoutWidget)
        self.gridLayout.setContentsMargins(0, 0, 0, 0)
        self.gridLayout.setObjectName("gridLayout")
        self.textBrowser_GIRFy_GY = QtWidgets.QTextBrowser(self.gridLayoutWidget)
        self.textBrowser_GIRFy_GY.setObjectName("textBrowser_GIRFy_GY")
        self.gridLayout.addWidget(self.textBrowser_GIRFy_GY, 2, 4, 1, 1)
        self.pushButton_openGIRFx_GZ = QtWidgets.QPushButton(self.gridLayoutWidget)
        self.pushButton_openGIRFx_GZ.setObjectName("pushButton_openGIRFx_GZ")
        self.gridLayout.addWidget(self.pushButton_openGIRFx_GZ, 1, 5, 1, 1)
        self.pushButton_openGIRFy_GZ = QtWidgets.QPushButton(self.gridLayoutWidget)
        self.pushButton_openGIRFy_GZ.setObjectName("pushButton_openGIRFy_GZ")
        self.gridLayout.addWidget(self.pushButton_openGIRFy_GZ, 2, 5, 1, 1)
        self.pushButton_openGIRFx_GY = QtWidgets.QPushButton(self.gridLayoutWidget)
        self.pushButton_openGIRFx_GY.setObjectName("pushButton_openGIRFx_GY")
        self.gridLayout.addWidget(self.pushButton_openGIRFx_GY, 1, 3, 1, 1)
        self.textBrowser_GIRFx_GX = QtWidgets.QTextBrowser(self.gridLayoutWidget)
        self.textBrowser_GIRFx_GX.setObjectName("textBrowser_GIRFx_GX")
        self.gridLayout.addWidget(self.textBrowser_GIRFx_GX, 1, 2, 1, 1)
        self.textBrowser_GIRFy_GZ = QtWidgets.QTextBrowser(self.gridLayoutWidget)
        self.textBrowser_GIRFy_GZ.setObjectName("textBrowser_GIRFy_GZ")
        self.gridLayout.addWidget(self.textBrowser_GIRFy_GZ, 2, 6, 1, 1)
        self.label_3 = QtWidgets.QLabel(self.gridLayoutWidget)
        self.label_3.setObjectName("label_3")
        self.gridLayout.addWidget(self.label_3, 0, 4, 1, 1, QtCore.Qt.AlignHCenter)
        self.textBrowser_GIRFx_GZ = QtWidgets.QTextBrowser(self.gridLayoutWidget)
        self.textBrowser_GIRFx_GZ.setObjectName("textBrowser_GIRFx_GZ")
        self.gridLayout.addWidget(self.textBrowser_GIRFx_GZ, 1, 6, 1, 1)
        self.pushButton_openGIRFy_GX = QtWidgets.QPushButton(self.gridLayoutWidget)
        self.pushButton_openGIRFy_GX.setObjectName("pushButton_openGIRFy_GX")
        self.gridLayout.addWidget(self.pushButton_openGIRFy_GX, 2, 1, 1, 1)
        self.label_6 = QtWidgets.QLabel(self.gridLayoutWidget)
        self.label_6.setObjectName("label_6")
        self.gridLayout.addWidget(self.label_6, 2, 0, 1, 1)
        self.pushButton_openGIRFx_GX = QtWidgets.QPushButton(self.gridLayoutWidget)
        self.pushButton_openGIRFx_GX.setObjectName("pushButton_openGIRFx_GX")
        self.gridLayout.addWidget(self.pushButton_openGIRFx_GX, 1, 1, 1, 1)
        self.label_5 = QtWidgets.QLabel(self.gridLayoutWidget)
        self.label_5.setObjectName("label_5")
        self.gridLayout.addWidget(self.label_5, 1, 0, 1, 1, QtCore.Qt.AlignHCenter|QtCore.Qt.AlignVCenter)
        self.textBrowser_GIRFx_GY = QtWidgets.QTextBrowser(self.gridLayoutWidget)
        self.textBrowser_GIRFx_GY.setObjectName("textBrowser_GIRFx_GY")
        self.gridLayout.addWidget(self.textBrowser_GIRFx_GY, 1, 4, 1, 1)
        self.textBrowser_GIRFy_GX = QtWidgets.QTextBrowser(self.gridLayoutWidget)
        self.textBrowser_GIRFy_GX.setObjectName("textBrowser_GIRFy_GX")
        self.gridLayout.addWidget(self.textBrowser_GIRFy_GX, 2, 2, 1, 1)
        self.pushButton_openGIRFy_GY = QtWidgets.QPushButton(self.gridLayoutWidget)
        self.pushButton_openGIRFy_GY.setObjectName("pushButton_openGIRFy_GY")
        self.gridLayout.addWidget(self.pushButton_openGIRFy_GY, 2, 3, 1, 1)
        self.label_2 = QtWidgets.QLabel(self.gridLayoutWidget)
        self.label_2.setObjectName("label_2")
        self.gridLayout.addWidget(self.label_2, 0, 2, 1, 1, QtCore.Qt.AlignHCenter)
        self.label_4 = QtWidgets.QLabel(self.gridLayoutWidget)
        self.label_4.setObjectName("label_4")
        self.gridLayout.addWidget(self.label_4, 0, 6, 1, 1, QtCore.Qt.AlignHCenter)
        self.label_39 = QtWidgets.QLabel(self.centralwidget)
        self.label_39.setGeometry(QtCore.QRect(30, 120, 22, 59))
        self.label_39.setObjectName("label_39")
        self.pushButton_CalKtra_nominal = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_CalKtra_nominal.setGeometry(QtCore.QRect(150, 490, 211, 71))
        self.pushButton_CalKtra_nominal.setObjectName("pushButton_CalKtra_nominal")
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(30, 360, 50, 12))
        self.label.setObjectName("label")
        self.label_41 = QtWidgets.QLabel(self.centralwidget)
        self.label_41.setGeometry(QtCore.QRect(190, 110, 42, 73))
        self.label_41.setObjectName("label_41")
        self.pushButton_SaveKtra_GIRF = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_SaveKtra_GIRF.setGeometry(QtCore.QRect(630, 650, 41, 31))
        self.pushButton_SaveKtra_GIRF.setObjectName("pushButton_SaveKtra_GIRF")
        self.pushButton_SaveKtra_nominal = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_SaveKtra_nominal.setGeometry(QtCore.QRect(230, 650, 41, 31))
        self.pushButton_SaveKtra_nominal.setObjectName("pushButton_SaveKtra_nominal")
        self.pushButton_OpenSeqFile = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_OpenSeqFile.setGeometry(QtCore.QRect(70, 50, 111, 23))
        self.pushButton_OpenSeqFile.setObjectName("pushButton_OpenSeqFile")
        self.pushButton_ShowKtraView_GIRF = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_ShowKtraView_GIRF.setGeometry(QtCore.QRect(590, 580, 131, 61))
        self.pushButton_ShowKtraView_GIRF.setObjectName("pushButton_ShowKtraView_GIRF")
        self.textBrowser_SeqFile = QtWidgets.QTextBrowser(self.centralwidget)
        self.textBrowser_SeqFile.setGeometry(QtCore.QRect(190, 50, 711, 22))
        self.textBrowser_SeqFile.setObjectName("textBrowser_SeqFile")
        self.label_40 = QtWidgets.QLabel(self.centralwidget)
        self.label_40.setGeometry(QtCore.QRect(100, 110, 42, 73))
        self.label_40.setObjectName("label_40")
        self.pushButton_ShowKtraView_nominal = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_ShowKtraView_nominal.setGeometry(QtCore.QRect(190, 580, 131, 61))
        self.pushButton_ShowKtraView_nominal.setObjectName("pushButton_ShowKtraView_nominal")
        self.listWidget_G_encode2 = QtWidgets.QListWidget(self.centralwidget)
        self.listWidget_G_encode2.setGeometry(QtCore.QRect(240, 110, 31, 61))
        self.listWidget_G_encode2.setSelectionRectVisible(True)
        self.listWidget_G_encode2.setObjectName("listWidget_G_encode2")
        item = QtWidgets.QListWidgetItem()
        self.listWidget_G_encode2.addItem(item)
        item = QtWidgets.QListWidgetItem()
        self.listWidget_G_encode2.addItem(item)
        item = QtWidgets.QListWidgetItem()
        self.listWidget_G_encode2.addItem(item)
        self.pushButton_ShowGIRFView = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_ShowGIRFView.setGeometry(QtCore.QRect(360, 410, 189, 51))
        self.pushButton_ShowGIRFView.setObjectName("pushButton_ShowGIRFView")
        self.gridLayoutWidget_2 = QtWidgets.QWidget(self.centralwidget)
        self.gridLayoutWidget_2.setGeometry(QtCore.QRect(30, 180, 891, 100))
        self.gridLayoutWidget_2.setObjectName("gridLayoutWidget_2")
        self.gridLayout_2 = QtWidgets.QGridLayout(self.gridLayoutWidget_2)
        self.gridLayout_2.setContentsMargins(0, 0, 0, 0)
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.label_14 = QtWidgets.QLabel(self.gridLayoutWidget_2)
        self.label_14.setObjectName("label_14")
        self.gridLayout_2.addWidget(self.label_14, 0, 2, 1, 1)
        self.doubleSpinBox_GY = QtWidgets.QDoubleSpinBox(self.gridLayoutWidget_2)
        self.doubleSpinBox_GY.setDecimals(4)
        self.doubleSpinBox_GY.setMaximum(10000000.0)
        self.doubleSpinBox_GY.setProperty("value", 1.0)
        self.doubleSpinBox_GY.setObjectName("doubleSpinBox_GY")
        self.gridLayout_2.addWidget(self.doubleSpinBox_GY, 1, 1, 1, 1)
        self.label_7 = QtWidgets.QLabel(self.gridLayoutWidget_2)
        self.label_7.setObjectName("label_7")
        self.gridLayout_2.addWidget(self.label_7, 0, 4, 1, 1)
        self.label_10 = QtWidgets.QLabel(self.gridLayoutWidget_2)
        self.label_10.setObjectName("label_10")
        self.gridLayout_2.addWidget(self.label_10, 1, 4, 1, 1)
        self.doubleSpinBox_GrampX = QtWidgets.QDoubleSpinBox(self.gridLayoutWidget_2)
        self.doubleSpinBox_GrampX.setDecimals(0)
        self.doubleSpinBox_GrampX.setMinimum(0.0)
        self.doubleSpinBox_GrampX.setMaximum(10000000.0)
        self.doubleSpinBox_GrampX.setProperty("value", 0.0)
        self.doubleSpinBox_GrampX.setObjectName("doubleSpinBox_GrampX")
        self.gridLayout_2.addWidget(self.doubleSpinBox_GrampX, 0, 3, 1, 1)
        self.label_9 = QtWidgets.QLabel(self.gridLayoutWidget_2)
        self.label_9.setObjectName("label_9")
        self.gridLayout_2.addWidget(self.label_9, 1, 0, 1, 1)
        self.doubleSpinBox_GrampY = QtWidgets.QDoubleSpinBox(self.gridLayoutWidget_2)
        self.doubleSpinBox_GrampY.setDecimals(0)
        self.doubleSpinBox_GrampY.setMinimum(0.0)
        self.doubleSpinBox_GrampY.setMaximum(10000000.0)
        self.doubleSpinBox_GrampY.setProperty("value", 0.0)
        self.doubleSpinBox_GrampY.setObjectName("doubleSpinBox_GrampY")
        self.gridLayout_2.addWidget(self.doubleSpinBox_GrampY, 1, 3, 1, 1)
        self.label_17 = QtWidgets.QLabel(self.gridLayoutWidget_2)
        self.label_17.setObjectName("label_17")
        self.gridLayout_2.addWidget(self.label_17, 2, 2, 1, 1)
        self.doubleSpinBox_GrampZ = QtWidgets.QDoubleSpinBox(self.gridLayoutWidget_2)
        self.doubleSpinBox_GrampZ.setDecimals(0)
        self.doubleSpinBox_GrampZ.setMinimum(0.0)
        self.doubleSpinBox_GrampZ.setMaximum(10000000.0)
        self.doubleSpinBox_GrampZ.setProperty("value", 0.0)
        self.doubleSpinBox_GrampZ.setObjectName("doubleSpinBox_GrampZ")
        self.gridLayout_2.addWidget(self.doubleSpinBox_GrampZ, 2, 3, 1, 1)
        self.label_15 = QtWidgets.QLabel(self.gridLayoutWidget_2)
        self.label_15.setObjectName("label_15")
        self.gridLayout_2.addWidget(self.label_15, 1, 2, 1, 1)
        self.doubleSpinBox_MaxVoltage = QtWidgets.QDoubleSpinBox(self.gridLayoutWidget_2)
        self.doubleSpinBox_MaxVoltage.setDecimals(0)
        self.doubleSpinBox_MaxVoltage.setMaximum(10000000.0)
        self.doubleSpinBox_MaxVoltage.setProperty("value", 20.0)
        self.doubleSpinBox_MaxVoltage.setObjectName("doubleSpinBox_MaxVoltage")
        self.gridLayout_2.addWidget(self.doubleSpinBox_MaxVoltage, 1, 5, 1, 1)
        self.label_16 = QtWidgets.QLabel(self.gridLayoutWidget_2)
        self.label_16.setObjectName("label_16")
        self.gridLayout_2.addWidget(self.label_16, 2, 0, 1, 1)
        self.doubleSpinBox_GX = QtWidgets.QDoubleSpinBox(self.gridLayoutWidget_2)
        self.doubleSpinBox_GX.setDecimals(4)
        self.doubleSpinBox_GX.setMinimum(0.0)
        self.doubleSpinBox_GX.setMaximum(10000000.0)
        self.doubleSpinBox_GX.setProperty("value", 1.0)
        self.doubleSpinBox_GX.setObjectName("doubleSpinBox_GX")
        self.gridLayout_2.addWidget(self.doubleSpinBox_GX, 0, 1, 1, 1)
        self.label_8 = QtWidgets.QLabel(self.gridLayoutWidget_2)
        self.label_8.setObjectName("label_8")
        self.gridLayout_2.addWidget(self.label_8, 0, 0, 1, 1)
        self.doubleSpinBox_MaxCurrent = QtWidgets.QDoubleSpinBox(self.gridLayoutWidget_2)
        self.doubleSpinBox_MaxCurrent.setDecimals(0)
        self.doubleSpinBox_MaxCurrent.setMaximum(10000000.0)
        self.doubleSpinBox_MaxCurrent.setProperty("value", 10.0)
        self.doubleSpinBox_MaxCurrent.setObjectName("doubleSpinBox_MaxCurrent")
        self.gridLayout_2.addWidget(self.doubleSpinBox_MaxCurrent, 0, 5, 1, 1)
        self.doubleSpinBox_GZ = QtWidgets.QDoubleSpinBox(self.gridLayoutWidget_2)
        self.doubleSpinBox_GZ.setDecimals(4)
        self.doubleSpinBox_GZ.setMaximum(10000000.0)
        self.doubleSpinBox_GZ.setProperty("value", 1.0)
        self.doubleSpinBox_GZ.setObjectName("doubleSpinBox_GZ")
        self.gridLayout_2.addWidget(self.doubleSpinBox_GZ, 2, 1, 1, 1)
        self.label_13 = QtWidgets.QLabel(self.centralwidget)
        self.label_13.setGeometry(QtCore.QRect(300, 140, 61, 20))
        self.label_13.setObjectName("label_13")
        self.doubleSpinBox_FOV = QtWidgets.QDoubleSpinBox(self.centralwidget)
        self.doubleSpinBox_FOV.setGeometry(QtCore.QRect(330, 140, 122, 20))
        self.doubleSpinBox_FOV.setDecimals(2)
        self.doubleSpinBox_FOV.setMinimum(0.0)
        self.doubleSpinBox_FOV.setMaximum(10000000.0)
        self.doubleSpinBox_FOV.setProperty("value", 10.0)
        self.doubleSpinBox_FOV.setObjectName("doubleSpinBox_FOV")
        self.radioButton_normalization = QtWidgets.QRadioButton(self.centralwidget)
        self.radioButton_normalization.setGeometry(QtCore.QRect(470, 140, 86, 16))
        self.radioButton_normalization.setObjectName("radioButton_normalization")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 942, 21))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        self.listWidget_G_read.setCurrentRow(0)
        self.listWidget_G_encode1.setCurrentRow(1)
        self.listWidget_G_encode2.setCurrentRow(2)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        __sortingEnabled = self.listWidget_G_read.isSortingEnabled()
        self.listWidget_G_read.setSortingEnabled(False)
        item = self.listWidget_G_read.item(0)
        item.setText(_translate("MainWindow", "x"))
        item = self.listWidget_G_read.item(1)
        item.setText(_translate("MainWindow", "y"))
        item = self.listWidget_G_read.item(2)
        item.setText(_translate("MainWindow", "z"))
        self.listWidget_G_read.setSortingEnabled(__sortingEnabled)
        self.pushButton_CalKtra_GIRF.setText(_translate("MainWindow", "Calcurate k-trajectory (GIRF)"))
        __sortingEnabled = self.listWidget_G_encode1.isSortingEnabled()
        self.listWidget_G_encode1.setSortingEnabled(False)
        item = self.listWidget_G_encode1.item(0)
        item.setText(_translate("MainWindow", "x"))
        item = self.listWidget_G_encode1.item(1)
        item.setText(_translate("MainWindow", "y"))
        item = self.listWidget_G_encode1.item(2)
        item.setText(_translate("MainWindow", "z"))
        self.listWidget_G_encode1.setSortingEnabled(__sortingEnabled)
        self.pushButton_openGIRFx_GZ.setText(_translate("MainWindow", "open"))
        self.pushButton_openGIRFy_GZ.setText(_translate("MainWindow", "open"))
        self.pushButton_openGIRFx_GY.setText(_translate("MainWindow", "open"))
        self.label_3.setText(_translate("MainWindow", "GY"))
        self.pushButton_openGIRFy_GX.setText(_translate("MainWindow", "open"))
        self.label_6.setText(_translate("MainWindow", "y_data"))
        self.pushButton_openGIRFx_GX.setText(_translate("MainWindow", "open"))
        self.label_5.setText(_translate("MainWindow", "x"))
        self.pushButton_openGIRFy_GY.setText(_translate("MainWindow", "open"))
        self.label_2.setText(_translate("MainWindow", "GX"))
        self.label_4.setText(_translate("MainWindow", "GZ"))
        self.label_39.setText(_translate("MainWindow", "read"))
        self.pushButton_CalKtra_nominal.setText(_translate("MainWindow", "Calcurate  k-trajectory (nominal)"))
        self.label.setText(_translate("MainWindow", "GIRF"))
        self.label_41.setText(_translate("MainWindow", "encode2"))
        self.pushButton_SaveKtra_GIRF.setText(_translate("MainWindow", "save"))
        self.pushButton_SaveKtra_nominal.setText(_translate("MainWindow", "save"))
        self.pushButton_OpenSeqFile.setText(_translate("MainWindow", "open sequence file"))
        self.pushButton_ShowKtraView_GIRF.setText(_translate("MainWindow", "k-trajectory view"))
        self.label_40.setText(_translate("MainWindow", "encode1"))
        self.pushButton_ShowKtraView_nominal.setText(_translate("MainWindow", "k-trajectory view"))
        __sortingEnabled = self.listWidget_G_encode2.isSortingEnabled()
        self.listWidget_G_encode2.setSortingEnabled(False)
        item = self.listWidget_G_encode2.item(0)
        item.setText(_translate("MainWindow", "x"))
        item = self.listWidget_G_encode2.item(1)
        item.setText(_translate("MainWindow", "y"))
        item = self.listWidget_G_encode2.item(2)
        item.setText(_translate("MainWindow", "z"))
        self.listWidget_G_encode2.setSortingEnabled(__sortingEnabled)
        self.pushButton_ShowGIRFView.setText(_translate("MainWindow", "GIRF View"))
        self.label_14.setText(_translate("MainWindow", "GX Ramp [us]"))
        self.label_7.setText(_translate("MainWindow", "Max Current [I]"))
        self.label_10.setText(_translate("MainWindow", "Max Voltage [V]"))
        self.label_9.setText(_translate("MainWindow", "GY [G/cm/A]"))
        self.label_17.setText(_translate("MainWindow", "GZ Ramp [us]"))
        self.label_15.setText(_translate("MainWindow", "GY Ramp [us]"))
        self.label_16.setText(_translate("MainWindow", "GZ [G/cm/A]"))
        self.label_8.setText(_translate("MainWindow", "GX [G/cm/A]"))
        self.label_13.setText(_translate("MainWindow", "FOV"))
        self.radioButton_normalization.setText(_translate("MainWindow", "-pi~pi"))
