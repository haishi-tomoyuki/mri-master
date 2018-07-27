# -*- coding: utf-8 -*-
"""
Created on Thu Jul 19 11:30:04 2018

@author: Ai Nakao
"""

import numpy as np
from scipy import fftpack
import matplotlib.pyplot as plt
import os
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import (QWidget, QPushButton, QLineEdit, QSizePolicy, QFileDialog, QApplication)
import sys
import convert_KtraCal
from Cal_ktrajectoryMainwindow import Ui_MainWindow
from LoadImage import *



class Cal_kloc(QtWidgets.QMainWindow):
    def __init__(self,parent=None):
        super().__init__(parent)
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        
        #シグナルスロットを実装
        self.ui.pushButton_OpenSeqFile.clicked.connect(self.Open_seqfile)
        self.ui.pushButton_openGIRFx_GX.clicked.connect(self.Open_GIRFx_GXfile)
        self.ui.pushButton_openGIRFy_GX.clicked.connect(self.Open_GIRFy_GXfile)
        self.ui.pushButton_openGIRFx_GY.clicked.connect(self.Open_GIRFx_GYfile)
        self.ui.pushButton_openGIRFy_GY.clicked.connect(self.Open_GIRFy_GYfile)
        self.ui.pushButton_openGIRFx_GZ.clicked.connect(self.Open_GIRFx_GZfile)
        self.ui.pushButton_openGIRFy_GZ.clicked.connect(self.Open_GIRFy_GZfile)
        
        self.ui.pushButton_ShowGIRFView.clicked.connect(self.ShowGIRFView)

        self.ui.pushButton_CalKtra_nominal.clicked.connect(self.CalKtra_nominal)
        self.ui.pushButton_CalKtra_GIRF.clicked.connect(self.CalKtra_GIRF)
        
        self.ui.pushButton_ShowKtraView_nominal.clicked.connect(self.ShowKtraView_nominal)
        self.ui.pushButton_ShowKtraView_GIRF.clicked.connect(self.ShowKtraView_GIRF)
        
        self.ui.pushButton_SaveKtra_nominal.clicked.connect(self.saveSeqFile_nominal)
        self.ui.pushButton_SaveKtra_GIRF.clicked.connect(self.saveSeqFile_GIRF)
        
    def Open_seqfile(self):
        self.filename_seqfile = self.openFileNameDialog()
        self.ui.textBrowser_SeqFile.setText(self.filename_seqfile)

        
    def Open_GIRFx_GXfile(self):
        self.filename_GIRFx_GX = self.openFileNameDialog()
        self.ui.textBrowser_GIRFx_GX.setText(self.filename_GIRFx_GX)
    
    def Open_GIRFy_GXfile(self):
        self.filename_GIRFy_GX = self.openFileNameDialog()
        self.ui.textBrowser_GIRFy_GX.setText(self.filename_GIRFy_GX)
        
    def Open_GIRFx_GYfile(self):
        self.filename_GIRFx_GY = self.openFileNameDialog()
        self.ui.textBrowser_GIRFx_GY.setText(self.filename_GIRFx_GY)
    
    def Open_GIRFy_GYfile(self):
        self.filename_GIRFy_GY= self.openFileNameDialog()
        self.ui.textBrowser_GIRFy_GY.setText(self.filename_GIRFy_GY)
    
    def Open_GIRFx_GZfile(self):
        self.filename_GIRFx_GZ = self.openFileNameDialog()
        self.ui.textBrowser_GIRFx_GZ.setText(self.filename_GIRFx_GZ)
    
    def Open_GIRFy_GZfile(self):
        self.filename_GIRFy_GZ = self.openFileNameDialog()
        self.ui.textBrowser_GIRFy_GZ.setText(self.filename_GIRFy_GZ)
        
    def ShowGIRFView(self):
        self.GIRFx_GX = np.fromfile(self.filename_GIRFx_GX)
        self.GIRFy_GX = np.fromfile(self.filename_GIRFy_GX, dtype="complex128")
        self.GIRFx_GY = np.fromfile(self.filename_GIRFx_GY)
        self.GIRFy_GY = np.fromfile(self.filename_GIRFy_GY, dtype="complex128")
        self.GIRFx_GZ = np.fromfile(self.filename_GIRFx_GZ)
        self.GIRFy_GZ = np.fromfile(self.filename_GIRFy_GZ, dtype="complex128")
        self.plot_GIRF()

    def GetGeneralParams(self):
        self.FOV = self.ui.doubleSpinBox_FOV.value()
        
        self.Gr = self.ui.listWidget_G_read.currentItem().text()
        self.Ge1 = self.ui.listWidget_G_encode1.currentItem().text()
        self.Ge2 = self.ui.listWidget_G_encode2.currentItem().text()
        
        self.hardware_GX = self.ui.doubleSpinBox_GX.value()
        self.hardware_GY = self.ui.doubleSpinBox_GY.value()
        self.hardware_GZ = self.ui.doubleSpinBox_GZ.value()
        
        self.hardware_GrampX = int(self.ui.doubleSpinBox_GrampX.value())
        self.hardware_GrampY = int(self.ui.doubleSpinBox_GrampY.value())
        self.hardware_GrampZ = int(self.ui.doubleSpinBox_GrampZ.value())
        
        self.hardware_MaxCurrent = self.ui.doubleSpinBox_MaxCurrent.value()
    
    def plot_GIRF(self):     
        fig = plt.figure()    
 
        ax1 = fig.add_axes((0.1, 0.1, 0.8, 0.3))
        ax2 = fig.add_axes((0.1, 0.4, 0.8, 0.3), sharex=ax1)
        ax3 = fig.add_axes((0.1, 0.7, 0.8, 0.3), sharex=ax1)

        ax2.tick_params(labelbottom="off")
        ax3.tick_params(labelbottom="off")
        ax1.set_xlabel("frequency [kHz]")
        
        ax1.set_ylabel("GX")
        ax2.set_ylabel("GY")
        ax3.set_ylabel("GZ")
        ax1.plot(self.GIRFx_GX,abs(self.GIRFy_GX), 'r-')
        ax2.plot(self.GIRFx_GY,abs(self.GIRFy_GY), 'b-')
        ax3.plot(self.GIRFx_GZ,abs(self.GIRFy_GZ), 'g-')
        #fig.tight_layout()  # タイトルとラベルが被るのを解消    
        plt.show()
        
    def CalKtra_nominal(self):
        self.GetGeneralParams()
        print("ok2")
        self.seq = SeqInfo(self.filename_seqfile, isSeqchart = True, GrampX=self.hardware_GrampX, GrampY=self.hardware_GrampY, GrampZ=self.hardware_GrampZ)
        self.kloc_x_nominal = Calc_kloc.Calc_kloc_nominal(self.seq.seq_GX, self.seq.seq_AD, I_max=self.hardware_MaxCurrent, G_eff=self.hardware_GX, DW=self.seq.DW, actualNR=self.seq.actualNR)
        self.kloc_x_nominal /= self.FOV
        self.kloc_y_nominal = Calc_kloc.Calc_kloc_nominal(self.seq.seq_GY, self.seq.seq_AD, I_max=self.hardware_MaxCurrent, G_eff=self.hardware_GY, DW=self.seq.DW, actualNR=self.seq.actualNR)
        self.kloc_y_nominal /= self.FOV
        self.kloc_z_nominal = Calc_kloc.Calc_kloc_nominal(self.seq.seq_GZ, self.seq.seq_AD, I_max=self.hardware_MaxCurrent, G_eff=self.hardware_GZ, DW=self.seq.DW, actualNR=self.seq.actualNR)
        self.kloc_z_nominal /= self.FOV
        if self.ui.radioButton_normalization.isChecked()==True:
            self.kloc_x_nominal = np.pi/(self.seq.nx_recon-self.seq.nx_recon/2)*self.kloc_x_nominal
            self.kloc_y_nominal = np.pi/(self.seq.nx_recon-self.seq.nx_recon/2)*self.kloc_y_nominal
            self.kloc_z_nominal = np.pi/(self.seq.nx_recon-self.seq.nx_recon/2)*self.kloc_z_nominal
        
    def CalKtra_GIRF(self):
        self.GetGeneralParams()
        self.seq = SeqInfo(self.filename_seqfile, isSeqchart = True, GrampX=self.hardware_GrampX, GrampY=self.hardware_GrampY, GrampZ=self.hardware_GrampZ)
        self.seq_GX_GIRF =  Calc_kloc.Calc_gradient_predict(self.filename_GIRFy_GX, self.filename_GIRFx_GX, self.seq.seq_GX, f_cut=20, amp_cut=1.2, offset=32768)
        self.seq_GY_GIRF =  Calc_kloc.Calc_gradient_predict(self.filename_GIRFy_GY, self.filename_GIRFx_GY, self.seq.seq_GY, f_cut=20, amp_cut=1.2, offset=32768)
        self.seq_GZ_GIRF =  Calc_kloc.Calc_gradient_predict(self.filename_GIRFy_GZ, self.filename_GIRFx_GZ, self.seq.seq_GZ, f_cut=20, amp_cut=1.2, offset=32768)
        self.kloc_x_GIRF = Calc_kloc.Calc_kloc_nominal(self.seq_GX_GIRF, self.seq.seq_AD, I_max = self.hardware_MaxCurrent, G_eff=self.hardware_GX, DW=self.seq.DW, actualNR=self.seq.actualNR)
        self.kloc_x_GIRF /= self.FOV
        self.kloc_y_GIRF = Calc_kloc.Calc_kloc_nominal(self.seq_GY_GIRF, self.seq.seq_AD, I_max = self.hardware_MaxCurrent, G_eff=self.hardware_GY, DW=self.seq.DW, actualNR=self.seq.actualNR)
        self.kloc_y_GIRF /= self.FOV
        self.kloc_z_GIRF = Calc_kloc.Calc_kloc_nominal(self.seq_GZ_GIRF, self.seq.seq_AD, I_max = self.hardware_MaxCurrent, G_eff=self.hardware_GZ, DW=self.seq.DW, actualNR=self.seq.actualNR)
        self.kloc_z_GIRF /= self.FOV
        if self.ui.radioButton_normalization.isChecked()==True:
            self.kloc_x_GIRF = np.pi/(self.seq.nx_recon-self.seq.nx_recon/2)*self.kloc_x_GIRF
            self.kloc_y_GIRF = np.pi/(self.seq.nx_recon-self.seq.nx_recon/2)*self.kloc_y_GIRF
            self.kloc_z_GIRF = np.pi/(self.seq.nx_recon-self.seq.nx_recon/2)*self.kloc_z_GIRF
        
        
    def saveSeqFile_nominal(self):
        filename = self.saveFileDialog()
        if filename != '':
            self.kloc_x_nominal.tofile(filename)
            
        filename = self.saveFileDialog()
        if filename != '':
            self.kloc_y_nominal.tofile(filename)
        filename = self.saveFileDialog()
        
        filename = self.saveFileDialog()  
        if filename != '':
            self.kloc_z_nominal.tofile(filename)

    def saveSeqFile_GIRF(self):
        filename = self.saveFileDialog()
        if filename != '':
            self.kloc_x_GIRF.tofile(filename)
            
        filename = self.saveFileDialog()
        if filename != '':
            self.kloc_y_GIRF.tofile(filename)
            
        filename = self.saveFileDialog()
        if filename != '':
            self.kloc_z_GIRF.tofile(filename)


    def openFileNameDialog(self):    
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        fileName, _ = QFileDialog.getOpenFileName(self,"QFileDialog.getOpenFileName()", "","All Files (*);;Seq Files (*.seq);;Text Files (*.txt)", options=options)
        if fileName:
            print(fileName)
        return fileName
    
    def saveFileDialog(self):    
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        fileName, _ = QFileDialog.getSaveFileName(self,"QFileDialog.getSaveFileName()","","All Files (*);;Seq Files (*.seq);;Text Files (*.txt)", options=options)
#        if fileName:
 #           print(fileName)
        return fileName
    
    
    
    def ShowKtraView_nominal(self):
        self.GetGeneralParams() 
        if self.Gr == "x":
            kloc_1 = self.kloc_x_nominal
        elif self.Gr == "y":
            kloc_1 = self.kloc_y_nominal
        elif self.Gr == "z": 
            kloc_1 = self.kloc_z_nominal
        
        if self.Ge1 == "x":
            kloc_2 = self.kloc_x_nominal
        elif self.Ge1 == "y":
            kloc_2 = self.kloc_y_nominal
        elif self.Ge1 == "z": 
            kloc_2 = self.kloc_z_nominal
             
        fig = plt.figure()
        ax1 = fig.add_axes((0.1, 0.1, 0.8, 0.8))
        ax1.plot(kloc_1,kloc_2)
        plt.show()
        
    def ShowKtraView_GIRF(self):
        if self.Gr == "x":
            kloc_1 = self.kloc_x_GIRF
        elif self.Gr == "y":
            kloc_1 = self.kloc_y_GIRF
        elif self.Gr == "z": 
            kloc_1 = self.kloc_z_GIRF
        
        if self.Ge1 == "x":
            kloc_2 = self.kloc_x_GIRF
        elif self.Ge1 == "y":
            kloc_2 = self.kloc_y_GIRF
        elif self.Ge1 == "z": 
            kloc_2 = self.kloc_z_GIRF
            
        fig = plt.figure()
        ax1 = fig.add_axes((0.1, 0.1, 0.8, 0.8))
        ax1.plot(kloc_1,kloc_2)
        plt.show()



            
               
def run_app():
    app = QtWidgets.QApplication(sys.argv)
    form = Cal_kloc()
    form.show()
    sys.exit(app.exec_()) 

if __name__ == '__main__':
    run_app()
        
