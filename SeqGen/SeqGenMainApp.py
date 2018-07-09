# -*- coding: utf-8 -*-
"""
Created on Mon Mar 12 17:31:48 2018

@author: Yasuhiko Terada
"""

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import (QWidget, QPushButton, QLineEdit, QSizePolicy, QFileDialog, QApplication)
import sys
import convert
from SeqGenMainwindow import Ui_MainWindow

import numpy as np
import matplotlib.pyplot as plt
sys.path.append('../Imager')
from LoadImage import Seqevent, Calc_seqchart
from SeqGen import *
import os
import re

        
class MyForm(QtWidgets.QMainWindow):
    def __init__(self,parent=None):
        super().__init__(parent)
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        
        #ここでシグナルスロットを実装
        self.ui.pushButton_genSeq.clicked.connect(self.GenSeq)
        self.ui.pushButton_open_Spinecho_H_tau.clicked.connect(self.Open_SEH_tau)
        self.ui.actionSaveSeq.triggered.connect(self.saveSeqFile)
        self.ui.actionOpen_ini_file.triggered.connect(self.LoadIniFileDialog)
        self.ui.actionSave_ini_file.triggered.connect(self.SaveIniFileDialog)
        self.ui.pushButton_ShowEventView.clicked.connect(self.showEventView)
        self.ui.pushButton_RFpulse90.clicked.connect(self.Open_RFpulse90)
        self.ui.pushButton_RFpulse180.clicked.connect(self.Open_RFpulse180)
        
        #外部で定義したクラスを使用
        self.hardware = Hardware()
        
        #設定ファイルの読み込みとバックアップ
        self.LoadIniFile('SeqGenMainwindow.ini')
        self.SaveIniFile('SeqGenMainwindow.ini.bak')
        
    def closeEvent(self, event):
        #設定ファイルの保存
        self.SaveIniFile('SeqGenMainwindow.ini')
        event.accept()
        
    def GenSeq(self):
        self.GetHardwareInfo()
        if self.ui.tabWidget_ImageParams.currentIndex() == 0:
            self.GenSeq_SE()
        if self.ui.tabWidget_ImageParams.currentIndex() == 1:
            self.GenSeq_SEH()
        
    def GenSeq_SE(self):
        #self.close
        TE = self.ui.doubleSpinBox_SE_TE.value()
        _is3D = self.ui.checkBox_SE_is3D.isChecked()
        if not hasattr(self,"SE"):          
            self.SE = SpinEcho(TE, self.hardware, is3D=_is3D)
        self.GetGeneralParams(self.SE)
        self.SE.TE = self.ui.doubleSpinBox_SE_TE.value()
        self.SE.is3D = self.ui.checkBox_SE_is3D.isChecked()
        self.SE.genSeq()
        self.SE.addComment(self.ui.textEdit_comments.toPlainText())
        self.SE.CheckCurrentLimit()
        self.showPulseList(self.SE)
        self.SE.pulse2event()
        #self.showEventList(self.SE)
        self.SE.gen_notes()
        self.showNotes(self.SE)
        self.RefreshGeneralParams(self.SE)

    def Open_SEH_tau(self):
        if not hasattr(self,"SEH"):
            self.SEH = SpinEcho_H(self.hardware)
        if self.ui.radioButton_ADevent.isChecked() == True:
            self.SEH.AD_event = True
        else:
            self.SEH.AD_event = False
        self.SEH.filename_tau = self.openFileNameDialog()
        self.SEH.fromfile(self.SEH.filename_tau)
        for tau in self.SEH.tau_list:
            self.ui.textBrowser_Spinecho_H_tau.append(str(tau))
            
    def Open_RFpulse90(self):
#        self.filename_RF90 = os.path.basename(self.openFileNameDialog())
        self.filename_RF90 = self.openFileNameDialog()
        self.ui.textBrowser_RFpulse90.setText(self.filename_RF90)

    def Open_RFpulse180(self):
#        self.filename_RF180 = os.path.basename(self.openFileNameDialog())
        self.filename_RF180 = self.openFileNameDialog()
        self.ui.textBrowser_RFpulse180.setText(self.filename_RF180)
            
    def GenSeq_SEH(self):
        if not hasattr(self,"SEH"):
            self.SEH = SpinEcho_H(self.hardware)
        self.filename_RF90 = self.ui.textBrowser_RFpulse90.toPlainText()
        self.filename_RF180 = self.ui.textBrowser_RFpulse180.toPlainText()
        self.GetGeneralParams(self.SEH)
        self.SEH.genSeq(filename_RF90 = self.filename_RF90, filename_RF180 = self.filename_RF180)
        self.SEH.addComment(self.ui.textEdit_comments.toPlainText())
        self.SEH.CheckCurrentLimit()
        self.showPulseList(self.SEH)
        self.SEH.pulse2event()
        #self.showEventList(self.seqDesign)
        self.SEH.gen_notes()
        self.showNotes(self.SEH)
        self.RefreshGeneralParams(self.SEH)
       
        
    def GetGeneralParams(self, seqDesign):
        seqDesign.NX = int(self.ui.spinBox_NX.value())
        seqDesign.NR = int(self.ui.spinBox_NR.value())
        seqDesign.N1 = int(self.ui.spinBox_N1.value())
        seqDesign.N2 = int(self.ui.spinBox_N2.value())
        seqDesign.S1 = int(self.ui.spinBox_S1.value())
        seqDesign.S2 = int(self.ui.spinBox_S2.value())
        seqDesign.DU = int(self.ui.spinBox_DU.value())
        seqDesign.TR = int(self.ui.spinBox_TR.value())
        seqDesign.DW = int(self.ui.spinBox_DW.value())
        seqDesign.SW = self.ui.doubleSpinBox_SW.value()
        seqDesign.OF = self.ui.doubleSpinBox_OF.value()
        seqDesign.FOVr = self.ui.doubleSpinBox_FOVr.value()
        seqDesign.FOVe1 = self.ui.doubleSpinBox_FOVe1.value()
        seqDesign.FOVe2 = self.ui.doubleSpinBox_FOVe2.value()
        seqDesign.Gr = self.ui.listWidget_G_read.currentItem().text()
        seqDesign.G1 = self.ui.listWidget_G_encode1.currentItem().text()
        seqDesign.G2 = self.ui.listWidget_G_encode2.currentItem().text()
        
        
        
    def RefreshGeneralParams(self, seqDesign):
        self.ui.spinBox_NX.setValue(seqDesign.NX)        
        self.ui.spinBox_NR.setValue(seqDesign.NR)        
        self.ui.spinBox_N1.setValue(seqDesign.N1)        
        self.ui.spinBox_N2.setValue(seqDesign.N2)        
        self.ui.spinBox_S1.setValue(seqDesign.S1)        
        self.ui.spinBox_S2.setValue(seqDesign.S2)        
        self.ui.spinBox_DU.setValue(seqDesign.DU)        
        self.ui.spinBox_TR.setValue(seqDesign.TR)        
        self.ui.spinBox_DW.setValue(seqDesign.DW)        
        self.ui.doubleSpinBox_SW.setValue(seqDesign.SW)        
        self.ui.doubleSpinBox_OF.setValue(seqDesign.OF)        
        self.ui.doubleSpinBox_FOVr.setValue(seqDesign.FOVr)  
        self.ui.doubleSpinBox_FOVe1.setValue(seqDesign.FOVe1)  
        self.ui.doubleSpinBox_FOVe2.setValue(seqDesign.FOVe2)
        
    def GetHardwareInfo(self):
        #tabWidget_Hardwareから必要な情報を取り出す
        self.hardware.GX = self.ui.doubleSpinBox_GX.value()
        self.hardware.GY = self.ui.doubleSpinBox_GY.value()
        self.hardware.GZ = self.ui.doubleSpinBox_GZ.value()
        self.hardware.MaxCurrent = self.ui.doubleSpinBox_MaxCurrent.value()
        self.hardware.MaxVoltage = self.ui.doubleSpinBox_MaxVoltage.value()
        self.hardware.GrampX = int(self.ui.doubleSpinBox_GrampX.value())
        self.hardware.GrampY = int(self.ui.doubleSpinBox_GrampY.value())
        self.hardware.GrampZ = int(self.ui.doubleSpinBox_GrampZ.value())
        self.hardware.resistance_x = self.ui.doubleSpinBox_resistance_x.value()
        self.hardware.resistance_y = self.ui.doubleSpinBox_resistance_y.value()
        self.hardware.resistance_z = self.ui.doubleSpinBox_resistance_z.value()
        self.hardware.GDA = int(self.ui.doubleSpinBox_GDA.value())
        
    def showPulseList(self, SeqDesign):    
        self.ui.textBrowser_history.setText('start, duration, type, value, option, gradType')
        for pulse in SeqDesign.pulse_RF:
            self.ui.textBrowser_history.append(str(pulse.t_start)+' '+str(pulse.duration)+' '+str(pulse.type)+' '+str(pulse.value)+' '+str(pulse.option))        
        for pulse in SeqDesign.pulse_GX:
            self.ui.textBrowser_history.append(str(pulse.t_start)+' '+str(pulse.duration)+' '+str(pulse.type)+' '+str(pulse.value)+' '+str(pulse.option)+' '+str(pulse.gradType))
        for pulse in SeqDesign.pulse_GY:
            self.ui.textBrowser_history.append(str(pulse.t_start)+' '+str(pulse.duration)+' '+str(pulse.type)+' '+str(pulse.value)+' '+str(pulse.option)+' '+str(pulse.gradType))
        for pulse in SeqDesign.pulse_GZ:
            self.ui.textBrowser_history.append(str(pulse.t_start)+' '+str(pulse.duration)+' '+str(pulse.type)+' '+str(pulse.value)+' '+str(pulse.option)+' '+str(pulse.gradType))
        for pulse in SeqDesign.pulse_AD:
            self.ui.textBrowser_history.append(str(pulse.t_start)+' '+str(pulse.duration)+' '+str(pulse.type)+' '+str(pulse.value)+' '+str(pulse.option))        
        
    def showEventList(self, SeqDesign):
        self.ui.textBrowser_history.append('')
        for event in SeqDesign.event_list:
            self.ui.textBrowser_history.append(event.line)
    
    def showNotes(self, SeqDesign):
        self.ui.textBrowser_seq.setText('')
        for note in SeqDesign.notes:
            self.ui.textBrowser_seq.append(note)
    
    def saveSeqFile(self):
        filename = self.saveFileDialog()
        #ここにseqファイルを保存する命令を書く
        if self.ui.tabWidget_ImageParams.currentIndex() == 0:
            if hasattr(self,"SE"):
                self.SE.SaveSeq(filename)
        if self.ui.tabWidget_ImageParams.currentIndex() == 1:
            if hasattr(self,"SEH"):
                self.SEH.SaveSeq(filename)
            
        
    def openFileNameDialog(self):    
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        fileName, _ = QFileDialog.getOpenFileName(self,"QFileDialog.getOpenFileName()", "","All Files (*);;Seq Files (*.seq);;Text Files (*.txt)", options=options)
        if fileName:
            print(fileName)
        return fileName
 
    def openFileNamesDialog(self):    
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        files, _ = QFileDialog.getOpenFileNames(self,"QFileDialog.getOpenFileNames()", "","All Files (*);;Seq Files (*.seq);;Text Files (*.txt)", options=options)
        if files:
            print(files)
        return files
          
    def saveFileDialog(self):    
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        fileName, _ = QFileDialog.getSaveFileName(self,"QFileDialog.getSaveFileName()","","All Files (*);;Seq Files (*.seq);;Text Files (*.txt)", options=options)
#        if fileName:
 #           print(fileName)
        return fileName
    
    """
    def seqchart(self, seqDesign):
        seqDesign.seq_GX = Calc_seqchart.Grad_simple(seqDesign.event_GX, seqDesign.TR, seqDesign.N1, seqDesign.N2, Gramp=seqDesign.hardware.GrampX)   
        seqDesign.seq_GY = Calc_seqchart.Grad_simple(seqDesign.event_GY, seqDesign.TR, seqDesign.N1, seqDesign.N2, Gramp=seqDesign.hardware.GrampY)
        seqDesign.seq_GZ = Calc_seqchart.Grad_simple(seqDesign.event_GZ, seqDesign.TR, seqDesign.N1, seqDesign.N2, Gramp=seqDesign.hardware.GrampZ)
        seqDesign.seq_RFx, seqDesign.seq_RFy = Calc_seqchart.RF(seqDesign.event_RF, seqDesign.TR)
        seqDesign.seq_AD = Calc_seqchart.AD(seqDesign.event_AD, seqDesign.TR, seqDesign.NR, seqDesign.DW)
    
    def plot_seqchart(self, seqDesign):
        fig = plt.figure()    
        
        #ax1 = fig.add_subplot(511)
        #ax2 = fig.add_subplot(512, sharex=ax1)
        #ax3 = fig.add_subplot(513, sharex=ax1)
        #ax4 = fig.add_subplot(514, sharex=ax1)
        #ax5 = fig.add_subplot(515, sharex=ax1)
        
        ax1 = fig.add_axes((0.15, 0.82, 0.8, 0.18))
        ax2 = fig.add_axes((0.15, 0.64, 0.8, 0.18), sharex=ax1)
        ax3 = fig.add_axes((0.15, 0.46, 0.8, 0.18), sharex=ax1)
        ax4 = fig.add_axes((0.15, 0.28, 0.8, 0.18), sharex=ax1)
        ax5 = fig.add_axes((0.15, 0.1, 0.8, 0.18), sharex=ax1)
        
        ax1.tick_params(labelbottom="off")
        ax2.tick_params(labelbottom="off")
        ax3.tick_params(labelbottom="off")
        ax4.tick_params(labelbottom="off")
        ax5.set_xlabel("Time [us]")
        ax1.set_ylabel("RF")
        ax2.set_ylabel("GX")
        ax3.set_ylabel("GY")
        ax4.set_ylabel("GZ")
        ax5.set_ylabel("AD")
        
        ax1.plot(seqDesign.seq_RFx[:,], 'r-')
        ax1.plot(seqDesign.seq_RFy[:,], 'b-')
        ax2.plot(seqDesign.seq_GX[:,0], 'r-')
        ax3.plot(seqDesign.seq_GY[:,0], 'b-')
        ax4.plot(seqDesign.seq_GZ[:,0], 'g-')
        ax5.plot(seqDesign.seq_AD[:,], 'y-')
        #fig.tight_layout()  # タイトルとラベルが被るのを解消   
        plt.show()
    """
    
    
    def showEventView(self):
       # print(vars(self.SE))
        
        if self.ui.tabWidget_ImageParams.currentIndex() == 0:
            if hasattr(self,"SE"):
                self.SE.Seqchart()
                self.SE.plot_seqchart()
                #self.seqchart(self.SE)
                #self.plot_seqchart(self.SE)
        
        if self.ui.tabWidget_ImageParams.currentIndex() == 1:
            if hasattr(self,"SEH"):
                self.SEH.Seqchart()
                self.SEH.plot_seqchart()
                #self.seqchart(self.SEH)
                #self.plot_seqchart(self.SEH)
                
                
    def LoadIniFileDialog(self):
        filename = self.openFileNameDialog()
        self.LoadIniFile(filename)
        
    def LoadIniFile(self, filename):
        with open(filename, 'rt') as f:
            lines = f.readlines() # 1行毎にファイル終端まで全て読む(改行文字も含まれる)
            
            for line in lines:
                if re.findall('NX=', line):
                    self.ui.spinBox_NX.setValue(int(line[3:]))
                if re.findall('NR=', line):
                    self.ui.spinBox_NR.setValue(int(line[3:]))
                if re.findall('N1=', line):
                    self.ui.spinBox_N1.setValue(int(line[3:]))
                if re.findall('N2=', line):
                    self.ui.spinBox_N2.setValue(int(line[3:]))
                if re.findall('S1=', line):
                    self.ui.spinBox_S1.setValue(int(line[3:]))
                if re.findall('S2=', line):
                    self.ui.spinBox_S2.setValue(int(line[3:]))
                if re.findall('DU=', line):
                    self.ui.spinBox_DU.setValue(int(line[3:]))
                if re.findall('TR=', line):
                    self.ui.spinBox_TR.setValue(int(line[3:]))
                if re.findall('DW=', line):
                    self.ui.spinBox_DW.setValue(int(line[3:]))
                if re.findall('SW=', line):
                    self.ui.doubleSpinBox_SW.setValue(float(line[3:]))
                if re.findall('OF=', line):
                    self.ui.doubleSpinBox_OF.setValue(float(line[3:]))
                if re.findall('FOVr=', line):
                    self.ui.doubleSpinBox_FOVr.setValue(float(line[5:]))
                if re.findall('FOVe1=', line):
                    self.ui.doubleSpinBox_FOVe1.setValue(float(line[6:]))
                if re.findall('FOVe2=', line):
                    self.ui.doubleSpinBox_FOVe2.setValue(float(line[6:]))

                
                if re.findall('GX=', line):
                    self.ui.doubleSpinBox_GX.setValue(float(line[3:]))
                if re.findall('GY=', line):
                    self.ui.doubleSpinBox_GY.setValue(float(line[3:]))
                if re.findall('GZ=', line):
                    self.ui.doubleSpinBox_GZ.setValue(float(line[3:]))
                
                if re.findall('GrampX=', line):
                    self.ui.doubleSpinBox_GrampX.setValue(float(line[7:]))
                if re.findall('GrampY=', line):
                    self.ui.doubleSpinBox_GrampY.setValue(float(line[7:]))
                if re.findall('GrampZ=', line):
                    self.ui.doubleSpinBox_GrampZ.setValue(float(line[7:]))
                
                if re.findall('MaxCurrent=', line):
                    self.ui.doubleSpinBox_MaxCurrent.setValue(float(line[11:]))
                if re.findall('MaxVoltage=', line):
                    self.ui.doubleSpinBox_MaxVoltage.setValue(float(line[11:]))
                
                if re.findall('resistance_x=', line):
                    self.ui.doubleSpinBox_resistance_x.setValue(float(line[13:]))
                if re.findall('resistance_y=', line):
                    self.ui.doubleSpinBox_resistance_y.setValue(float(line[13:]))
                if re.findall('resistance_z=', line):
                    self.ui.doubleSpinBox_resistance_z.setValue(float(line[13:]))
                
                if re.findall('GDA=', line):
                    self.ui.doubleSpinBox_GDA.setValue(float(line[4:]))
                
                if re.findall('RF90=', line):
                    self.ui.textBrowser_RFpulse90.setText(line[5:len(line)-1])
                if re.findall('RF180=', line):
                    self.ui.textBrowser_RFpulse180.setText(line[6:len(line)-1])
                    
    def SaveIniFileDialog(self):
        filename = self.saveFileDialog()
        self.SaveIniFile(filename)
        
    def SaveIniFile(self, filename):
        with open(filename, 'wt') as f:
            f.write('[Parameters]\n')
            f.write('NX='+str(self.ui.spinBox_NX.value())+'\n')
            f.write('NR='+str(self.ui.spinBox_NR.value())+'\n')
            f.write('N1='+str(self.ui.spinBox_N1.value())+'\n')
            f.write('N2='+str(self.ui.spinBox_N2.value())+'\n')
            f.write('S1='+str(self.ui.spinBox_S1.value())+'\n')
            f.write('S2='+str(self.ui.spinBox_S2.value())+'\n')
            f.write('DU='+str(self.ui.spinBox_DU.value())+'\n')
            f.write('TR='+str(self.ui.spinBox_TR.value())+'\n')
            f.write('DW='+str(self.ui.spinBox_DW.value())+'\n')
            f.write('SW='+str(self.ui.doubleSpinBox_SW.value())+'\n')
            f.write('OF='+str(self.ui.doubleSpinBox_OF.value())+'\n')
            f.write('FOVr='+str(self.ui.doubleSpinBox_FOVr.value())+'\n')
            f.write('FOVe1='+str(self.ui.doubleSpinBox_FOVe1.value())+'\n')
            f.write('FOVe2='+str(self.ui.doubleSpinBox_FOVe2.value())+'\n')
            f.write('\n')
            f.write('[Gradient]\n')
            f.write('GX='+str(self.ui.doubleSpinBox_GX.value())+'\n')
            f.write('GY='+str(self.ui.doubleSpinBox_GY.value())+'\n')
            f.write('GZ='+str(self.ui.doubleSpinBox_GZ.value())+'\n')
            f.write('GrampX='+str(int(self.ui.doubleSpinBox_GrampX.value()))+'\n')
            f.write('GrampY='+str(int(self.ui.doubleSpinBox_GrampY.value()))+'\n')
            f.write('GrampZ='+str(int(self.ui.doubleSpinBox_GrampZ.value()))+'\n')
            f.write('MaxCurrent='+str(self.ui.doubleSpinBox_MaxCurrent.value())+'\n')
            f.write('MaxVoltage='+str(self.ui.doubleSpinBox_MaxVoltage.value())+'\n')
            f.write('resistance_x='+str(self.ui.doubleSpinBox_resistance_x.value())+'\n')
            f.write('resistance_y='+str(self.ui.doubleSpinBox_resistance_y.value())+'\n')
            f.write('resistance_z='+str(self.ui.doubleSpinBox_resistance_z.value())+'\n')
            f.write('GDA=' +str(int(self.ui.doubleSpinBox_GDA.value()))+'\n')
            
            f.write('\n')
            f.write('[RF pulse]\n')
            f.write('RF90='+self.ui.textBrowser_RFpulse90.toPlainText()+'\n')
            f.write('RF180='+str(self.ui.textBrowser_RFpulse180.toPlainText())+'\n')
            
            


                    
def run_app():
    app = QtWidgets.QApplication(sys.argv)
    form = MyForm()
    form.show()
    sys.exit(app.exec_()) 

if __name__ == '__main__':
    run_app()