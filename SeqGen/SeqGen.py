# -*- coding: utf-8 -*-
"""
Created on Tue May 29 18:57:17 2018

@author: Terada
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('../Imager')
from LoadImage import Seqevent, Calc_seqchart,Calc_kloc
import enum        #列挙型
import operator    #クラスのソート用
import csv
import re

class SeqDesign:
    def __init__(self, hardware):
        self.hardware = hardware
        #default value(共通パラメタ)
        self.FOVr = 6.4     #cm
        self.FOVe1 = 6.4
        self.FOVe2 = 6.4
        self.NX = 1
        self.NR = 128
        self.N1 = 128
        self.N2 = 128
        self.S1 = 256
        self.S2 = 256
        self.DU = 10
        self.SW = 5         #mm
        self.OF = [0]
        self.TR = 200       #ms
        self.DW = 10        #us
        
        self.Gr = 'x'
        self.G1 = 'y'
        self.G2 = 'z'
        
        self.Init()
    
    def Init(self):
        self.pulse_GX = []   #list of SeqPulse
        self.pulse_GY = []  
        self.pulse_GZ = []  
        self.pulse_RF = []   #including SeqPulse.type = RF, PH, TF
        self.pulse_AD = []
        
        self.notes = []      #これを最終的にファイル出力する
        
        self.event_GX = []   #list of Seqevent
        self.event_GY = []
        self.event_GZ = []
        self.event_RF = []
        self.event_AD = []
        self.event_list = []  #list of all seqevents
        
        self.comments = []   #list of comments
        
        self.isCurrentOver = None   #抵抗を考慮した電流制限値を超えたかどうか
    
    #RF pulse関係
    def addRF90(self, t_start, phase=0):
        seqPulseRF = SeqPulseRF(t_start, RFType.hard90, phase=phase)
        self.pulse_RF.append(seqPulseRF)
        return seqPulseRF
    def addRF180(self, t_start, phase=0):
        seqPulseRF = SeqPulseRF(t_start, RFType.hard180, phase=phase)
        self.pulse_RF.append(seqPulseRF)
        return seqPulseRF
    """
    def addRFpulse(self, seqPulseRF_default, t_start=-1):
        seqPulseRF = copy.copy(seqPulseRF_default)
        if t_start >= 0:
            seqPulseRF.t_start = int(t_start)
        self.pulse_RF.append(seqPulseRF)
        return seqPulseRF
    """
    def addRFpulse(self, t_start, rfType, duration=1000, phase=0, BW=-1, filename=None, RFpulseShapeX=None, RFpulseShapeY=None, comment=None):
        seqPulseRF = SeqPulseRF(t_start, rfType, duration=duration, phase=phase, BW=BW, filename=filename, RFpulseShapeX=RFpulseShapeX, RFpulseShapeY=RFpulseShapeY, comment=comment)
        self.pulse_RF.append(seqPulseRF)
        return seqPulseRF
    #AD 関係    
    def addAD(self, t_start, duration):
        seqPulseAD = SeqPulseAD(t_start, duration)
        self.pulse_AD.append(seqPulseAD)
        return seqPulseAD
    
    #Gradient関係
    def addReadGrad(self, t_start, duration, isNegative = False):
        Gr_HEX = self.calc_ReadGradAmp(self.Gr, isNegative = isNegative)
        seqPulseGrad = SeqPulseGrad(t_start, Gr_HEX, duration, self.Gr, gradType=GradType.Read)
        self.addGrad_sub(seqPulseGrad, self.Gr)
        return seqPulseGrad             
    def addPE1Grad(self, t_start, option='<-e5', table=None, filename=None):
        d_enc = self.calc_PhaseEncodeTime(self.G1, self.S1, self.FOVe1)
        #seqPulseGrad = SeqPulseGrad(t_start, self.S1*self.N1/2+32768, d_enc, self.G1, option=option, table=None, filename=None, gradType=GradType.PhaseEncode1)
        seqPulseGrad = SeqPulseGrad(t_start, 32768, d_enc, self.G1, option=option, table=None, filename=None, gradType=GradType.PhaseEncode1)
        self.addGrad_sub(seqPulseGrad, self.G1)
        return seqPulseGrad             
    def addPERew1Grad(self, t_start, option='<-c5', table=None, filename=None):
        d_enc = self.calc_PhaseEncodeTime(self.G1, self.S1, self.FOVe1)
        #seqPulseGrad = SeqPulseGrad(t_start, 32768-self.S1*self.N1/2, d_enc, self.G1, option=option, table=None, filename=None, gradType=GradType.PhaseRewind1)
        seqPulseGrad = SeqPulseGrad(t_start, 32768, d_enc, self.G1, option=option, table=None, filename=None, gradType=GradType.PhaseRewind1)
        self.addGrad_sub(seqPulseGrad, self.G1)
        return seqPulseGrad             
    def addPE2Grad(self, t_start, option='<-e6', table=None, filename=None):
        d_enc = self.calc_PhaseEncodeTime(self.G2, self.S2, self.FOVe2)
        #seqPulseGrad = SeqPulseGrad(t_start, self.S2*self.N2/2+32768, d_enc, self.G2, option=option, table=None, filename=None, gradType=GradType.PhaseEncode2)
        seqPulseGrad = SeqPulseGrad(t_start, 32768, d_enc, self.G2, option=option, table=None, filename=None, gradType=GradType.PhaseEncode2)
        self.addGrad_sub(seqPulseGrad, self.G2)
        return seqPulseGrad             
    def addPERew2Grad(self, t_start, option='<-c6'):
        d_enc = self.calc_PhaseEncodeTime(self.G2, self.S2, self.FOVe2, table=None, filename=None)
        #seqPulseGrad = SeqPulseGrad(t_start, 32768-self.S2*self.N2/2, d_enc, self.G2, option=option, table=None, filename=None, gradType=GradType.PhaseRewind2)
        seqPulseGrad = SeqPulseGrad(t_start, 32768, d_enc, self.G2, option=option, table=None, filename=None, gradType=GradType.PhaseRewind2)
        self.addGrad_sub(seqPulseGrad, self.G2)
        return seqPulseGrad             
    def addSliceGrad(self, t_start, duration, BW=-1, isNegative = False):
        Gs_HEX = self.calc_SliceGradAmp(self.G2, BW=BW, isNegative = isNegative)
        if isNegative == False:
            seqPulseGrad = SeqPulseGrad(t_start, Gs_HEX, duration, self.G2, option='(GSlice_P)', gradType=GradType.Slice)
        else:
            seqPulseGrad = SeqPulseGrad(t_start, Gs_HEX, duration, self.G2, option='(GSlice_N)', gradType=GradType.Slice)
        self.addGrad_sub(seqPulseGrad, self.G2)
        return seqPulseGrad
    def addCrusherGrad(self, t_start, G_HEX, duration, direction):
        #direction : self.Gr, self.G1, self.G2
        seqPulseGrad = SeqPulseGrad(t_start, G_HEX, duration, direction, gradType=GradType.Crusher)
        self.addGrad_sub(seqPulseGrad, direction)
        return seqPulseGrad
                             
    def addGrad_sub(self, seqPulseGrad, direction):
        if direction == 'x':
            self.pulse_GX.append(seqPulseGrad)
        if direction == 'y':
            self.pulse_GY.append(seqPulseGrad)
        if direction == 'z':
            self.pulse_GZ.append(seqPulseGrad)  
    
    def getGradEff(self, direction): #Grad_efficiency [G/cm/A]
        if direction == 'x':
            G = self.hardware.GX     #Grad_efficiency [G/cm/A]
        if direction == 'y':
            G = self.hardware.GY
        if direction == 'z':
            G = self.hardware.GZ
        return G
    def getGradRamp(self, direction): 
        if direction == 'x':
            Gramp = self.hardware.GrampX     #Grad_efficiency [G/cm/A]
        if direction == 'y':
            Gramp = self.hardware.GrampY
        if direction == 'z':
            Gramp = self.hardware.GrampZ
        return Gramp
        
    def calc_ReadGradAmp(self, direction, isNegative = False):
        Gr = self.getGradEff(direction)
        current = 1/(self.DW*1e-6)/(Gr*1e-4*1e2)/42.58e6/(self.FOVr*1e-2)   
        if isNegative:
            current *= -1      
        return self.Current2HEX(current)  #HEX
    
    def calc_SliceGradAmp(self, direction, BW = -1, isNegative = False):
        if BW == -1:  #不明の場合
            return self.Current2HEX(0)         
        Gr = self.getGradEff(direction)
        current = BW / 42.58e6 / (Gr*1e-4*1e2) / (self.SW*1e-3)      
        if isNegative:
            current *= -1        
        return self.Current2HEX(current)  #[A]
    
    def calc_PhaseEncodeTime(self, direction, step_height, FOV_cm):  #[us]
        Ge = self.getGradEff(direction)
        return 1/FOV_cm/42.58e6/(step_height*self.hardware.MaxCurrent/32768)/(Ge*1e-4) * 1e6 #us       
            
    def Current2HEX(self, current):
        return current / (self.hardware.MaxCurrent*2) * 65536 + 32768
        
    def pulse2event(self):
        """
        pulse_GX -> seq_GX [list of seqEvent]などへの変換
        """
        self.event_GX = self.pulse2event_sub(self.pulse_GX)
        self.event_GY = self.pulse2event_sub(self.pulse_GY)
        self.event_GZ = self.pulse2event_sub(self.pulse_GZ)
        self.event_RF = self.pulse2event_sub(self.pulse_RF)
        self.event_AD = self.pulse2event_sub(self.pulse_AD)

        self.event_list = self.event_RF + self.event_GX + self.event_GY + self.event_GZ + self.event_AD
        self.event_list.sort(key=operator.attrgetter('time'))  #time順にsortする
        
    def pulse2event_sub(self, pulse_l):
        """
        pulse_GX -> event_GX などへの変換
        input pulse_l : list of seqPulse
        output event_l : list of seqEvent
        """
        event_l = []
        
        pulse_l.sort(key=operator.attrgetter('t_start'))  #t_start順にsortする
        
        val_init ='%.4X' % 0x8000
        phase_init = -1  #わざと変な値を入れて、RFの場合に最初必ずPHが入るようにしておく
        
        for pulse in pulse_l:
            if pulse.comment is None:
                line = self.format_time(pulse.t_start) + ' ' + pulse.type + ' ' + pulse.value + pulse.option
            else:
                line = self.format_time(pulse.t_start) + ' ' + pulse.type + ' ' + pulse.value + pulse.option + '\t' + ';'+ pulse.comment
            seqevent = Seqevent(line,self.N1,self.N2,self.S1,self.S2)
            seqevent.isTimeEvent = True
            seqevent.filename = pulse.filename
            event_l.append(seqevent)
            
            if pulse.type == 'GX' or pulse.type == 'GY' or pulse.type == 'GZ':  #立下り命令
                line = self.format_time(pulse.t_start+pulse.duration) + ' ' + pulse.type + ' ' + val_init
                seqevent = Seqevent(line,self.N1,self.N2,self.S1,self.S2)
                seqevent.isTimeEvent = True
                event_l.append(seqevent)
            
            
            if pulse.type == 'RF':  #RFの場合にはphaseをチェックする
                if pulse.phase != phase_init:
                    phase_val = int(pulse.phase/90*0x0100)
                    line = self.format_time(pulse.t_start-10) + ' PH ' + '%.4X' % phase_val
                    phase_init = phase_val
                    seqevent = Seqevent(line,self.N1,self.N2,self.S1,self.S2)
                    seqevent.isTimeEvent = True
                    event_l.append(seqevent)
            
            
        #時刻順にソートしておく
        event_l.sort(key=operator.attrgetter('time')) 
        
        #Gradイベントリストの後処理
        if len(pulse_l)>0:
            if pulse.type == 'GX' or pulse.type == 'GY' or pulse.type == 'GZ':
                for i in range(len(event_l)-1):
                    if i<len(event_l)-1:
                        if event_l[i].time == event_l[i+1].time:  #パルスの立下りと次のパルスの立ち上がりが同時刻イベントの場合
                            event_l.pop(i)
        
        return event_l
    
    def format_time(self, time):
        """
        input : time [us] (int)
        output : '00.001.000.0'
        """
        temp = str(time).zfill(8)
        return temp[0:2] + '.' + temp[2:5] + '.' + temp[5:8] + '.0'
    
    def gen_notes(self):
        self.notes = []
        
        #parameterを書き込む
        self.notes.append(':NX '+str(self.NX))
        self.notes.append(':NR '+str(self.NR))
        self.notes.append(':N1 '+str(self.N1))
        self.notes.append(':N2 '+str(self.N2))
        self.notes.append(':S1 '+str(self.S1))
        self.notes.append(':S2 '+str(self.S2))
        self.notes.append(':DU '+str(self.DU))
        self.notes.append(':TR '+str(self.TR))
        self.notes.append(':DW '+str(self.DW))
        self.notes.append('')
        self.notes.append(':SW '+str(self.SW))
        for i in range(len(self.OF)):
            self.notes.append(':OF '+str(self.OF[i]))
        
        self.notes.append('')
        
        #event_listを書き込む        
        for event in self.event_list:
            self.notes.append(event.line)
            
        self.notes.append('')

        #commentを書き込む
        self.comments.append(';SeqDesign.type='+self.type)
        self.notes += self.comments
    
    def addComment(self, comment):
        self.comments.append(';' + comment)
        
    #display用
    def showPulseList(self):    
        print ('start, duration, type, value, option, gradType, comment')
        for pulse in self.pulse_RF:
            print(pulse.t_start, pulse.duration, pulse.type, pulse.value, pulse.option, pulse.comment)
        for pulse in self.pulse_GX:
            print(pulse.t_start, pulse.duration, pulse.type, pulse.value, pulse.option, pulse.gradType, pulse.comment)
        for pulse in self.pulse_GY:
            print(pulse.t_start, pulse.duration, pulse.type, pulse.value, pulse.option, pulse.gradType, pulse.comment)
        for pulse in self.pulse_GZ:
            print(pulse.t_start, pulse.duration, pulse.type, pulse.value, pulse.option, pulse.gradType, pulse.comment)
        for pulse in self.pulse_AD:
            print(pulse.t_start, pulse.duration, pulse.type, pulse.value, pulse.option, pulse.comment)
            
    def showEventList(self):
        print ('')
        for event in self.event_list:
            print(event.line)
    
    def showNotes(self):
        print('')
        for note in self.notes:
            print(note)
            
    def Seqchart(self):
        self.seq_GX = Calc_seqchart.Grad_simple(self.event_GX, self.TR, self.N1, self.N2, Gramp=self.hardware.GrampX)
        self.seq_GY = Calc_seqchart.Grad_simple(self.event_GY, self.TR, self.N1, self.N2, Gramp=self.hardware.GrampY)
        self.seq_GZ = Calc_seqchart.Grad_simple(self.event_GZ, self.TR, self.N1, self.N2, Gramp=self.hardware.GrampZ)
        self.seq_RFx, self.seq_RFy = Calc_seqchart.RF(self.event_RF, self.TR)
        self.seq_AD = Calc_seqchart.AD(self.event_AD, self.TR, self.NR, self.DW)
    
    def CheckCurrentLimit(self):
        """
        抵抗を考慮して電流制限をチェックする
        pulse_GX, pulse_GY, pulse_GZ
        """
        isViolated_GX = self.CheckCurrentLimit_sub(self.pulse_GX, self.hardware.resistance_x, self.hardware.MaxCurrent, self.hardware.MaxVoltage)
        isViolated_GY = self.CheckCurrentLimit_sub(self.pulse_GY, self.hardware.resistance_y, self.hardware.MaxCurrent, self.hardware.MaxVoltage)
        isViolated_GZ = self.CheckCurrentLimit_sub(self.pulse_GZ, self.hardware.resistance_z, self.hardware.MaxCurrent, self.hardware.MaxVoltage)
        self.isCurrentOver = isViolated_GX or isViolated_GY or isViolated_GZ
        return self.isCurrentOver
        
    def CheckCurrentLimit_sub(self, pulse_GX, resistance, MaxCurrent, MaxVoltage):
        LimitCurrent = min(MaxCurrent, MaxVoltage/resistance)
        isViolated = False
        for pulse in pulse_GX:
            if pulse.gradType == GradType.Read or pulse.gradType == GradType.Slice:
                current = abs((int(pulse.value,16)-32768) / 32768 * MaxCurrent)
            if pulse.gradType == GradType.PhaseEncode1:
                current = self.N1 * self.S1 / 65536 * MaxCurrent
            if pulse.gradType == GradType.PhaseEncode2:
                current = self.N2 * self.S2 / 65536 * MaxCurrent
            if current > LimitCurrent:
                isViolated = True
                pulse.comment = 'Caution : Current exceeds limit!!!'
        return isViolated
            
    def CheckTR(self):
        """
        pulse終了時間よりもTRを長くする
        """
        self.CheckTR_sub(self.pulse_GX)
        self.CheckTR_sub(self.pulse_GY)
        self.CheckTR_sub(self.pulse_GZ)
        self.CheckTR_sub(self.pulse_RF)
        self.CheckTR_sub(self.pulse_AD)

    def CheckTR_sub(self, pulse_l):
        if len(pulse_l)>0:
            for pulse in pulse_l:
                last_time = int((pulse.t_start + pulse.duration)/1000 + 1)  #ms
                if last_time > self.TR:
                    print('TR is too short!!! Automatically modified TR')
                    self.TR = last_time
       
    def Cal_kloc(self):
        #ノミナルのkloc算出
        self.kloc_x_nominal = Calc_kloc.Calc_kloc_nominal(self.seq_GX, self.seq_AD, I_max=self.hardware.MaxCurrent, G_eff = self.hardware.GX, DW=self.DW, actualNR=self.NR, OS = self.OS)
        self.kloc_y_nominal = Calc_kloc.Calc_kloc_nominal(self.seq_GY, self.seq_AD, I_max=self.hardware.MaxCurrent, G_eff = self.hardware.GY, DW=self.DW, actualNR=self.NR, OS = self.OS)
        self.kloc_z_nominal = Calc_kloc.Calc_kloc_nominal(self.seq_GZ, self.seq_AD, I_max=self.hardware.MaxCurrent, G_eff = self.hardware.GZ, DW=self.DW, actualNR=self.NR, OS = self.OS)
        self.kloc_x_nominal.tofile("kloc_x_nominal.dbl")
        self.kloc_y_nominal.tofile("kloc_y_nominal.dbl")
        self.kloc_z_nominal.tofile("kloc_z_nominal.dbl")
        #GIRFのkloc算出
        filename_GIRF = "GIRF_GxB000.clx"
        filename_GIRFx = "GIRF_GxB000_x.dbl"
        self.kloc_x_GIRF = Calc_kloc.Calc_gradient_predict(filename_GIRF,filename_GIRFx, self.seq_GX, f_cut=20, amp_cut=1.2, offset=32768)        
        self.kloc_y_GIRF = Calc_kloc.Calc_gradient_predict(filename_GIRF,filename_GIRFx, self.seq_GY, f_cut=20, amp_cut=1.2, offset=32768)
        self.kloc_z_GIRF = Calc_kloc.Calc_gradient_predict(filename_GIRF,filename_GIRFx, self.seq_GZ, f_cut=20, amp_cut=1.2, offset=32768)
        self.kloc_x_GIRF.tofile("kloc_x_GIRF.dbl")
        self.kloc_y_GIRF.tofile("kloc_y_GIRF.dbl")
        self.kloc_z_GIRF.tofile("kloc_z_GIRF.dbl")
            
    def SaveSeq(self, filename):
        if filename != '':
            f = open(filename, 'w') # 書き込みモードで開く
            for line in self.notes:
                f.write(line+'\n')
            f.close()
    
        
    def LoadIniFile(self,filename):
        with open(filename, 'rt') as f:
            lines = f.readlines() # 1行毎にファイル終端まで全て読む(改行文字も含まれる)
            
            for line in lines:
                if re.findall('NX=', line):
                    self.NX=int(line[3:])
                if re.findall('NR=', line):
                    self.NR=int(line[3:])
                if re.findall('N1=', line):
                    self.N1=int(line[3:])
                if re.findall('N2=', line):
                    self.N2=int(line[3:])
                if re.findall('S1=', line):
                    self.S1=int(line[3:])
                if re.findall('S2=', line):
                    self.S2=int(line[3:])
                if re.findall('DU=', line):
                    self.DU=int(line[3:])
                if re.findall('TR=', line):
                    self.TR=int(line[3:])
                if re.findall('DW=', line):
                    self.DW=int(line[3:])
                if re.findall('SW=', line):
                    self.SW=float(line[3:])
                if re.findall('OF=', line):
                    self.OF=[float(line[3:])]
                
                if re.findall('GX=', line):
                    self.hardware.GX=float(line[3:])
                if re.findall('GY=', line):
                    self.hardware.GY=float(line[3:])
                if re.findall('GZ=', line):
                    self.hardware.GZ=float(line[3:])
                
                if re.findall('GrampX=', line):
                    self.hardware.GrampX=float(line[7:])
                if re.findall('GrampY=', line):
                    self.hardware.GrampY=float(line[7:])
                if re.findall('GrampZ=', line):
                    self.hardware.GrampZ=float(line[7:])

                if re.findall('MaxCurrent=', line):
                    self.hardware.MaxCurrent=float(line[11:])
                if re.findall('MaxVoltage=', line):
                    self.hardware.MaxVoltage=float(line[11:])

                if re.findall('resistance_x=', line):
                    self.hardware.resistance_x=float(line[13:])
                if re.findall('resistance_y=', line):
                    self.hardware.resistance_y=float(line[13:])
                if re.findall('resistance_z=', line):
                    self.hardware.resistance_z=float(line[13:])                    


    def SaveIniFile(self,filename):
        with open(filename, 'wt') as f:
            f.write('[Parameters]\n')
            f.write('NX='+int(self.NX)+'\n')
            f.write('NR='+int(self.NR)+'\n')
            f.write('N1='+int(self.N1)+'\n')
            f.write('N2='+int(self.N2)+'\n')
            f.write('S1='+int(self.S1)+'\n')
            f.write('S2='+int(self.S2)+'\n')
            f.write('DU='+int(self.DU)+'\n')
            f.write('TR='+int(self.TR)+'\n')
            f.write('DW='+int(self.DW)+'\n')
            f.write('SW='+float(self.SW)+'\n')
            f.write('OF='+float(self.OF[0])+'\n')
            f.write('\n')
            f.write('[Gradient]\n')
            f.write('GX='+float(self.hardware.GX)+'\n')
            f.write('GY='+float(self.hardware.GY)+'\n')
            f.write('GZ='+float(self.hardware.GZ)+'\n')
            f.write('GrampX='+float(self.hardware.GrampX)+'\n')
            f.write('GrampY='+float(self.hardware.GrampY)+'\n')
            f.write('GrampZ='+float(self.hardware.GrampZ)+'\n')
            f.write('MaxCurrent='+float(self.hardware.MaxCurrent)+'\n')
            f.write('MaxVoltage='+float(self.hardware.MaxVoltage)+'\n')
            f.write('resistance_x='+float(self.hardware.resistance_x)+'\n')
            f.write('resistance_y='+float(self.hardware.resistance_y)+'\n')
            f.write('resistance_z='+float(self.hardware.resistance_z)+'\n')
            f.write('\n')
    def plot_seqchart(self):     
        fig = plt.figure()    
        """
        ax1 = fig.add_subplot(511)
        ax2 = fig.add_subplot(512, sharex=ax1)
        ax3 = fig.add_subplot(513, sharex=ax1)
        ax4 = fig.add_subplot(514, sharex=ax1)
        ax5 = fig.add_subplot(515, sharex=ax1)
        """
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
        ax1.plot(self.seq_RFx[:,], 'r-')
        ax1.plot(self.seq_RFy[:,], 'b-')
        ax2.plot(self.seq_GX[:,0], 'r-')
        ax3.plot(self.seq_GY[:,0], 'b-')
        ax4.plot(self.seq_GZ[:,0], 'g-')
        ax5.plot(self.seq_AD[:,], 'y-')
        #fig.tight_layout()  # タイトルとラベルが被るのを解消    
        plt.show()
        
    def Oblique(self, axis, angle1, angle2):
        """
        axis = 'X' or 'Y' or 'Z'
        angle1, angle2 : degree (right-handed system)
        transfer pulse_GX, pulse_GY, pulse_GZ, to oblique plane
        """
        None
 
                
class SpinEcho(SeqDesign):
    """
    TE : ms
    """
    def __init__(self, TE_ms, hardware, is3D = True, GDA = [0], \
                 filename_RF90 = None, filename_RF180 = None, \
                 duration_RF90 = 1000, duration_RF180 = 1000, \
                 BW_RF90 = -1, BW_RF180 = -1):
        super().__init__(hardware)
        self.type = 'SE'
        self.TE = TE_ms
        self.is3D = is3D
        self.hardware.GDA = GDA   #refocus gradient duration の微調整量[us] :
        
        self.filename_RF90 = filename_RF90
        self.filename_RF180 = filename_RF180
        self.duration_RF90 = duration_RF90
        self.duration_RF180 = duration_RF180
        self.BW_RF90 = BW_RF90
        self.BW_RF180 = BW_RF180
        
        
    def genSeq(self):
        self.Init()
        
        self.comments.append(';FOVr[cm]='+ str(self.FOVr))
        self.comments.append(';FOVe1[cm]='+ str(self.FOVe1))
        self.comments.append(';FOVe2[cm]='+ str(self.FOVe2))
        
        #Grampの取得
        GRampRead = self.getGradRamp(self.Gr)
        GRampEnc1 = self.getGradRamp(self.G1)
        GRampEnc2 = self.getGradRamp(self.G2)

        te = self.TE * 1e3
        t_90 = 5000     #start time (RF90) [us]
        d_180 = 120      #duration (RF180)
        t_180 =  t_90 + te / 2  - d_180/2      #start time (RF180)
        d_AD = self.DW * self.NR
        t_AD_c = t_90 + te              #center time (ADC)
        t_AD = t_AD_c - d_AD/2       #start time (ADC)
        if self.is3D:
            if self.filename_RF90 is not None:
                seqPulseRF90 = self.addRFpulse(t_90, RFType.UD, filename = self.filename_RF90, \
                                               duration = self.duration_RF90, BW = self.BW_RF90)
            else:
                seqPulseRF90 = self.addRF90(t_90)
            if self.filename_RF180 is not None:
                seqPulseRF180 = self.addRFpulse(t_180, RFType.UD, filename = self.filename_RF180, \
                                                duration = self.duration_RF180, BW = self.BW_RF180)
            else:
                seqPulseRF180 = self.addRF180(t_180, phase=90)
                
        else:  #2D
            if self.filename_RF90 is not None:
                seqPulseRF90 = self.addRFpulse(t_90, RFType.MS90UD, filename = self.filename_RF90, \
                                               duration = self.duration_RF90, BW = self.BW_RF90)
            else:         
                seqPulseRF90 = self.addRFpulse(t_90, RFType.MS90, phase=0)
            if self.filename_RF180 is not None:
                seqPulseRF180 = self.addRFpulse(t_180, RFType.MS180UD, filename = self.filename_RF180, \
                                                duration = self.duration_RF180, BW = self.BW_RF180)
            else:
                seqPulseRF180 = self.addRF180(t_180)
            #slice G
            t_SliceG_s = t_90-500
            d_SliceG =  seqPulseRF90.duration+1000
            t_SliceG_e = t_SliceG_s + d_SliceG
            self.addSliceGrad(t_SliceG_s, d_SliceG, BW=seqPulseRF90.BW)  
            #refocus gradient durationの計算
            t_90_c = t_90 + seqPulseRF90.duration/2
            #RFパルスの中心時刻とslice_Gradientの印加終了時刻の差 * 1.04 + 調整時間
            refocusGrad_duration = int((t_SliceG_e - t_90_c) * 1.04) + self.hardware.GDA[0]    
            self.addSliceGrad(t_SliceG_e, refocusGrad_duration, BW=seqPulseRF90.BW, isNegative=True)
            #crusher
            self.addCrusherGrad(t_180-GRampEnc2-1500, 0xC000, 500, self.G2)
            self.addCrusherGrad(t_180+seqPulseRF180.duration+300, 0xC000, 500, self.G2)
        
        #direction : self.Gr, self.G1, self.G2
        self.addAD(t_AD, d_AD)
        #read (dephase)
        d_Read_dephase = d_AD/2 + GRampRead/2
        t_Read_s = t_90 + seqPulseRF90.duration + 100
        self.addReadGrad(t_Read_s, d_Read_dephase)
        #read (rephase)
        t_Read_s = t_AD_c - d_AD/2 - GRampRead
        self.addReadGrad(t_Read_s, d_AD + GRampRead + 1000)
        #1st encode
        t_enc1 = t_180 + seqPulseRF180.duration + 100  #margin 100us
        self.addPE1Grad(t_enc1)
        #2nd encode
        if self.is3D:
            t_enc2 = t_180 + seqPulseRF180.duration + 100  #margin 100us
            self.addPE2Grad(t_enc2)
        
        self.CheckTR()

class GradientEcho(SeqDesign):
    """
    TE : ms
    """
    def __init__(self, TE_ms, hardware, is3D = True, GDA = [0], \
                 filename_RF90 = None, duration_RF90 = 1000, BW_RF90 = -1):
        super().__init__(hardware)
        self.type = 'GE'
        self.TE = TE_ms
        self.is3D = is3D
        self.hardware.GDA = GDA   #refocus gradient duration の微調整量[us] :
        self.filename_RF90 = filename_RF90
        self.duration_RF90 = duration_RF90
        self.BW_RF90 = BW_RF90
        
    def genSeq(self):
        self.Init()
        
        self.comments.append(';FOVr[cm]='+ str(self.FOVr))
        self.comments.append(';FOVe1[cm]='+ str(self.FOVe1))
        self.comments.append(';FOVe2[cm]='+ str(self.FOVe2))
        
        #Grampの取得
        GRampRead = self.getGradRamp(self.Gr)
        GRampEnc1 = self.getGradRamp(self.G1)
        GRampEnc2 = self.getGradRamp(self.G2)

        te = self.TE * 1e3              #[us]
        t_90 = 5000     #start time (RF90) [us]

        d_AD = self.DW * self.NR
        t_AD_c = t_90 + te              #center time (ADC)
        t_AD = t_AD_c - d_AD/2       #start time (ADC)
        if self.is3D:
            if self.filename_RF90 is not None:
                seqPulseRF90 = self.addRFpulse(t_90, RFType.UD, \
                            filename = self.filename_RF90, duration = self.duration_RF90,\
                            BW = self.BW_RF90)
            else:
                seqPulseRF90 = self.addRF90(t_90)
        else:  #2D
            if self.filename_RF90 is not None:
                seqPulseRF90 = self.addRFpulse(t_90, RFType.MS90UD, \
                            filename = self.filename_RF90, duration = self.duration_RF90,\
                            BW = self.BW_RF90)
            else:
                seqPulseRF90 = self.addRFpulse(t_90, RFType.MS90, phase=0)
            #slice G
            t_SliceG_s = t_90-500
            d_SliceG =  seqPulseRF90.duration+1000
            t_SliceG_e = t_SliceG_s + d_SliceG
            self.addSliceGrad(t_SliceG_s, d_SliceG, BW=seqPulseRF90.BW)  
            #refocus gradient durationの計算
            t_90_c = t_90 + seqPulseRF90.duration/2
            #RFパルスの中心時刻とslice_Gradientの印加終了時刻の差 * 1.04 + 調整時間
            refocusGrad_duration = int((t_SliceG_e - t_90_c) * 1.04) + self.hardware.GDA[0]    
            self.addSliceGrad(t_SliceG_e, refocusGrad_duration, BW=seqPulseRF90.BW, isNegative=True)
            
        #direction : self.Gr, self.G1, self.G2
        self.addAD(t_AD, d_AD)
        #read (dephase)
        d_Read_dephase = d_AD/2 + GRampRead/2
        t_Read_s = t_AD - d_AD/2 - GRampRead*3/2
        self.addReadGrad(t_Read_s, d_Read_dephase, isNegative=True)
        #read (rephase)
        t_Read_s = t_AD - GRampRead
        self.addReadGrad(t_Read_s, d_AD + GRampRead + 1000)
        #1st encode
        t_enc1 = t_90 + seqPulseRF90.duration + 100  #margin 100us
        self.addPE1Grad(t_enc1)
        #2nd encode
        if self.is3D:
            t_enc2 = t_90 + seqPulseRF90.duration + 100  #margin 100us
            self.addPE2Grad(t_enc2)
        
        self.CheckTR()

class SpinEcho_H(SeqDesign):
    def __init__(self, hardware):
        super().__init__(hardware)
        self.type = 'SEH'
        self.AD_renzoku = True
        
    def fromfile(self, filename):
        #ファイルからτとエクセルファイルからもらう
        """
        tau_list:１，τ1，...
    
        """
        self.tau_list = []
        with open(filename, 'r') as f:
            reader = csv.reader(f)
            next(reader)
            for row in reader:
                 self.tau_list.append(int(row[1])) 
    
    def genSeq(self, filename_RF90 = 'hard90_10us.txt', filename_RF180 = 'hard180_10us_y.txt'):
        self.Init()
        
        t_margin = 100
        self.N1 = 1
        self.N2 = 1
        self.NR = int((np.array(self.tau_list).sum() + t_margin + 100)/self.DW)
        
        t_90 = 5000     #start time (RF90) [us]
        seqPulseRF180_dummy = SeqPulseRF(0,RFType.UD,filename = filename_RF180)
        d_180 = seqPulseRF180_dummy.duration     #duration (RF180)
        t_180_temp = t_90
        t_AD_c = t_90
        self.addRFpulse(t_90, RFType.UD, filename=filename_RF90)
        d_AD = self.DW * self.NR
        if self.AD_renzoku == True:
            t_AD = t_90 - t_margin    
            self.addAD(t_AD, d_AD)
            for i in range(len(self.tau_list)):
                t_180 = t_180_temp + int(self.tau_list[i]/2) - d_180/2
                t_180_temp = t_180_temp + self.tau_list[i]
                self.addRFpulse(t_180, RFType.UD, filename = filename_RF180)

        else:
            for i in range(len(self.tau_list)):
                t_180 = t_180_temp + int(self.tau_list[i]/2) - d_180/2
                t_180_temp = t_180_temp + self.tau_list[i]
                self.addRFpulse(t_180, RFType.UD, filename = filename_RF180)
                t_AD_c = t_AD_c + self.tau_list[i]              #center time (ADC)
                t_AD = t_AD_c - d_AD/2       #start time (ADC)
                self.addAD(t_AD, d_AD)
                
        self.CheckTR()
  
class SeqPulse:
    def __init__(self, _type, t_start, value, duration=100, option='', table=None, filename=None, comment=None):
        self.type = _type    #GX, GY, GZ, AD, RF, PH
        self.t_start = int(t_start)         #us    
        self.value = '%.4X' % int(value)           #16進数
        self.duration = int(duration)       #us
        self.option = option                #<-v5{8000, ...}
        self.table = table                  #[8000,...]
        self.filename = filename            #'c:\...'
        self.comment = comment              #''
        
class SeqPulseRF(SeqPulse):
    def __init__(self, t_start, rfType, duration=120, phase=0, BW=-1, filename=None, RFpulseShapeX=None, RFpulseShapeY=None, comment=None):
        super().__init__('RF', t_start, 0, duration=duration, filename=filename, comment=comment)
        """
        t_start
        rfType
        duration
        phase
        filename
        RFpulseShapeX, RFPulseShapeY
        comment
        """
        if rfType == RFType.sinc90_8kHz:
            value = 0x0000
            self.FA = 90
            self.duration = 1000
            self.BW = 8000
        if rfType == RFType.sinc180_8kHz:
            value = 0x0001
            self.FA = 180
            self.duration = 1000
            self.BW = 8000
        if rfType == RFType.hard90:
            value = 0x0002
            self.FA = 90
            self.duration = 120
            self.BW = 1/120e-6
        if rfType == RFType.hard180:
            value = 0x0003
            self.FA = 180
            self.duration = 120
            self.BW = 1/120e-6
        if rfType == RFType.sinc90_4kHz:
            value = 0x0004
            self.FA = 90
            self.duration = 1000
            self.BW = 4000
        if rfType == RFType.sinc180_4kHz:
            value = 0x0005
            self.FA = 180
            self.duration = 1000
            self.BW = 4000
        if rfType == RFType.sinc90_4kHz_PI:
            value = 0x0006
            self.FA = 90
            self.duration = 500
            self.BW = 4000
        if rfType == RFType.sinc180_4kHz_PI:
            value = 0x0007
            self.FA = 180
            self.duration = 500
            self.BW = 4000
        if rfType == RFType.MS90:
            value = 0x000D
            self.FA = 90
            self.duration = 1000
            self.BW = 8000
        if rfType == RFType.MS180:
            value = 0x000E
            self.FA = 180
            self.duration = 1000
            self.BW = 8000
        if rfType == RFType.UD:
            value = 0x000F
            self.duration = duration
            self.option = '=['+filename+']'
            self.BW = BW
        if rfType == RFType.MS90UD:
            value = 0x000D
            self.option = '=['+filename+']'
            self.duration = duration
            self.BW = BW
        if rfType == RFType.MS180UD:
            value = 0x000E
            self.duration = duration
            self.option = '=['+filename+']'
            self.BW = BW
            
        self.value = '%.4X' % int(value)  
        self.rfType = rfType
        self.phase = phase
        self.RFpulseShapeX = RFpulseShapeX
        self.RFpulseShapeY = RFpulseShapeY
        
        if filename is not None:
            self.LoadRFPulse(filename)  #self.durationとself.BWが書き換えられる
        
    def LoadRFPulse(self, filename):
        if filename != '':
            try:
                f = open(filename)  
            except OSError as err:
                print("OS error: {0}".format(err))
                return -1
            else:
                lines = f.readlines() # 1行毎にファイル終端まで全て読む(改行文字も含まれる)
                f.close()
                
                #self.rfType = RFType.UD    #これどうする
                self.filename = filename
                self.option = '=['+filename+']'
                
                temp = lines[1].split('\t')
                Npoints = int(lines[0])
                ReadoutDwell =  int(int(temp[0]) *0.1)                          #us
                self.duration = Npoints * ReadoutDwell                #duration[us]
                amp = float(temp[1])  #増幅率
                self.BW = float(temp[2])*1000                           #BW[Hz]
                self.RFpulseShapeX = np.ones((self.duration))  #1us刻み
                self.RFpulseShapeY = np.zeros((self.duration))  #1us刻み
                for i in range(len(lines)-2):
                    temp = lines[i+2].split('\t')
                    self.RFpulseShapeX[i*ReadoutDwell:(i+1)*ReadoutDwell] = int(temp[0])*amp
                    self.RFpulseShapeY[i*ReadoutDwell:(i+1)*ReadoutDwell] = int(temp[1])*amp
                
                      
class SeqPulseAD(SeqPulse):
    def __init__(self, t_start, duration, comment=None):
        super().__init__('AD', t_start, 0x8000, duration=duration, comment=comment)

class SeqPulseGrad(SeqPulse):
    def __init__(self, t_start, value, duration, direction, option='', table=None, filename=None, comment=None, gradType = None):
        if direction == 'x':
            super().__init__('GX', t_start, value, duration=duration, option=option, table=table, filename=filename)
        if direction == 'y':
            super().__init__('GY', t_start, value, duration=duration, option=option, table=table, filename=filename)
        if direction == 'z':
            super().__init__('GZ', t_start, value, duration=duration, option=option, table=table, filename=filename)
        self.gradType = gradType
    
class GradType(enum.Enum):  #GradTypeを列挙型で定義しておく
    #この情報は、後でオブリークするときに必要になるはず
    Read = 'Read'
    Rephase = 'Rephase'
    Dephase = 'Dephase'
    PhaseEncode1 = 'PhaseEncode1'
    PhaseEncode2 = 'PhaseEncode2'
    PhaseRewind1 = 'PhaseRewind1'
    PhaseRewind2 = 'PhaseRewind2'
    Slice = 'Slice'
    Crusher = 'Crusher'

class RFType(enum.Enum):  #RFPuplseを列挙型で定義しておく
    sinc90_8kHz = 0
    sinc180_8kHz = 1
    hard90 = 2
    hard180 = 3
    sinc90_4kHz = 4
    sinc180_4kHz = 5
    sinc90_4kHz_PI = 6
    sinc180_4kHz_PI = 7
    MS90 = 8
    MS180 = 9
    UD = 10
    MS90UD = 11
    MS180UD = 12
    
class Hardware():
    def __init__(self):
        self.GX = 0.5   #G/cm/A
        self.GY = 0.5   #G/cm/A
        self.GZ = 0.5   #G/cm/A
        self.MaxCurrent = 10    #10 means from -10 to 10 V
        self.GrampX = 0 #us
        self.GrampY = 0 #us
        self.GrampZ = 0 #us
        self.GDA = [0] #gradientのパルス幅調整量 [us] (list型)
        
        self.MaxVoltage = 20    #[V]
        self.resistance_x = 1   #resistance [Ohm]
        self.resistance_y = 1
        self.resistance_z = 1
    



def test_SE():
    hardware = Hardware()
#    se = SpinEcho(10, hardware, is3D=False)
#    se = SpinEcho(10, hardware, is3D=False, \
#        filename_RF90 = 'test.txt', filename_RF180 = 'test180.txt', duration_RF90 = 1000, duration_RF180 = 1000,\
#        BW_RF90 = -1, BW_RF180 = -1)
    se = SpinEcho(10, hardware, is3D=False, filename_RF90 = 'sinc_2ms_2kHz.txt')
    se.DW = 10
    se.NR = 256
    se.TR = 50
    se.OF = [-10,0,10]
    se.genSeq()
    se.addComment('This is a test')
    se.showPulseList()
    se.pulse2event()
    se.CheckCurrentLimit()
    #se.showEventList()
    se.gen_notes()
    se.showNotes()
    se.Seqchart()
    #se.SaveSeq('spinecho.seq')
    """
    plt.plot(se.seq_RFx)
    plt.plot(se.seq_RFy)
    plt.plot(se.seq_AD)
    plt.plot(se.seq_GX[:,0])
    plt.plot(se.seq_GY[:,0])
    plt.plot(se.seq_GZ[:,0])
    """
    se.plot_seqchart()

def test_SEH():
    hardware = Hardware()
    SEH = SpinEcho_H(hardware)
    SEH.fromfile('tau.csv')
    SEH.DW=1
    SEH.TR=1000
    SEH.genSeq(filename_RF90 = 'hard90_10us.txt', filename_RF180 = 'hard180_10us_y.txt')
    SEH.showPulseList()
    SEH.pulse2event()
    SEH.CheckCurrentLimit()
    SEH.gen_notes()
    SEH.showNotes()
    SEH.Seqchart()
    """
    fig = plt.figure()
    plt.plot(SEH.seq_RFx)
    plt.plot(SEH.seq_RFy)
    plt.plot(SEH.seq_AD)
    plt.show()
    """
    SEH.plot_seqchart()

def test_GE():
    hardware = Hardware()
    hardware.GrampX = 100 #us
    hardware.GrampY = 100 #us
    hardware.GrampZ = 100 #us
#    ge = GradientEcho(10, hardware, is3D=False)
#    ge = GradientEcho(10, hardware, is3D=False, filename_RF90 = 'sinc_2ms_2kHz.txt',\
#                      duration_RF90 = 1000, BW_RF90 = -1)
    ge = GradientEcho(10, hardware, is3D=False, filename_RF90 = 'sinc_2ms_2kHz.txt')
    ge.DW = 10
    ge.NR = 256
    ge.TR = 50
    ge.genSeq()
    ge.addComment('This is a test')
    ge.showPulseList()
    ge.pulse2event()
    ge.CheckCurrentLimit()
    #ge.showEventList()
    ge.gen_notes()
    ge.showNotes()
    ge.Seqchart()
    #ge.Savegeq('Gradientecho.geq')
    ge.plot_seqchart()
    
if __name__ == '__main__':
    test_SE()
    #test_SEH()
    #test_GE()
    



