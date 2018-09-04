# -*- coding: utf-8 -*-
"""
Created on Sat Jun 30 15:29:31 2018

@author: Terada

Ver. 1
"""
import numpy as np
import matplotlib.pyplot as plt

class EPG:
    def __init__(self, NumOfColumns, m0=1, t1=0.5, t2=0.1):
        self.NumOfColumns = NumOfColumns    #CS matrixの行数
        self.setParams(m0, t1, t2)
              
    def setParams(self, m0, t1, t2):
        self.m0 = m0
        self.T1 = t1                #[s]
        self.T2 = t2                #[s]
        self.init_CS(self.NumOfColumns)
        
    def init_CS(self, NumOfColumns):
        self.CS0 = np.zeros((3,NumOfColumns),dtype=complex) #行 : F+, F-, Zに対応,  列 : kz
        self.CS0[2,0]= np.complex(self.m0, 0)
        self.CS = self.CS0.copy()
        
    def gen_Tmatrix(self, phi, alpha):
        T = np.zeros((3,3),dtype=complex)
        T[0][0] = np.complex((np.cos(alpha/2))**2, 0)
        T[0][1] = np.complex( np.cos(2*phi)*(np.sin(alpha/2))**2, np.sin(2*phi)*(np.sin(alpha/2))**2)
        T[0][2] = complex( np.sin(phi)*np.sin(alpha), -np.cos(phi)*np.sin(alpha))
        T[1][0] = complex( np.cos(2*phi)*(np.sin(alpha/2))**2, -np.sin(2*phi)*(np.sin(alpha/2))**2)
        T[1][1] = T[0][0] 
        T[1][2] = complex( np.sin(phi)*np.sin(alpha), np.cos(phi)*np.sin(alpha))
        T[2][0] = -T[1][2]/2
        T[2][1] = -T[0][2]/2
        T[2][2] = complex(np.cos(alpha), 0)
        return T
    
    def op_T(self,Tmatrix):
        self.CSm = self.CS.copy()
        self.CS = np.dot(Tmatrix, self.CSm)

    def op_S1(self): #shift operator (Δk=1)
        self.CSm = self.CS.copy()
        N = self.NumOfColumns
        
        #check
        if self.CS[0][N-1] != 0:
            print('number of colums for CS matrix is not sufficient!!!')
        
        self.CS[1][N-1] = np.complex(0,0)
        for i in range(N-1):
            self.CS[1][i] = self.CSm[1][i+1]
        self.CS[0][0] = np.conj(self.CS[1][0])
        for i in range(1,N):
            self.CS[0][i] = self.CSm[0][i-1]
            
    def op_Sm1(self): #shift operator (Δk=-1)
        self.CSm = self.CS.copy()
        N = self.NumOfColumns
        
        #check
        if self.CS[1][N-1] != 0:
            print('number of colums for CS matrix is not sufficient!!!')

        self.CS[0][N-1] = np.complex(0,0)
        for i in range(N-1):
            self.CS[0][i] = self.CSm[0][i+1]
        self.CS[1][0] = np.conj(self.CS[0][0])
        for i in range(1,N):
            self.CS[1][i] = self.CSm[1][i-1]
            
    def op_Relax(self, tau):  
        #tau: time interval [s]
        self.CSm = self.CS.copy()
        E1 = np.exp(-tau/self.T1)
        E2 = np.exp(-tau/self.T2)
        Ematrix = np.zeros((3,3))
        Ematrix[0][0] = E2
        Ematrix[1][1] = E2
        Ematrix[2][2] = E1
        self.CS = np.dot(Ematrix, self.CSm)    
        self.CS[2][0] += np.complex(self.m0*(1-E1), 0)
        
    def op_AD(self):
        return self.CS[0,0]
            
    """
    高級操作
    """
    def op_delay(self,tau):
        self.op_Relax(tau)
    
    def op_Gx(self, tau, polarity):
        self.op_Relax(tau)
        
        if polarity>0:
            self.op_S1()
        else:
            self.op_Sm1()
            
    def display_results(self):
        fig = plt.figure()
        plt.plot(self.t_echoPeak.real, "o")
        plt.plot(self.t_echoPeak.imag, "o")
        fig.show()
            
class EPG_CPMG(EPG):
    def __init__(self, ETL=10, FA_excite=90, FA_refocus=60, TE=20e-3):
        super().__init__(ETL*2+1)
        self.ETL = ETL              #echo train lengths
        self.FA_excite = FA_excite  #degree
        self.FA_refocus = FA_refocus  #degree
        self.TE = TE
        
    def run(self, m0=1, t1=0.5, t2=0.1):
        self.setParams(m0, t1, t2)
        self.RF1 = self.gen_Tmatrix(np.pi/2, self.FA_excite/180*np.pi)  #90y
        self.RF2 = self.gen_Tmatrix(0, self.FA_refocus/180*np.pi)    #alpha_x
        
        self.t_echoPeak = np.zeros((self.ETL), dtype = complex)
        
        #start
        self.op_T(self.RF1)
        
        for i in range(self.ETL):
            self.op_Gx(self.TE/2, 1)
            self.op_T(self.RF2)
            self.op_Gx(self.TE/2, 1)
            self.t_echoPeak[i] = self.op_AD()

class EPG_FISP(EPG):
    def __init__(self, ETL=10, FA=30, TR=30e-3):
        super().__init__(ETL+1)
        self.ETL=ETL
        self.FA=FA
        self.TR=TR
    def run(self, m0=1, t1=0.5, t2=0.1):
        self.setParams(m0, t1, t2)
        self.RF1 = self.gen_Tmatrix(0, self.FA/180*np.pi)  #90y
        
        self.t_echoPeak = np.zeros((self.ETL), dtype = complex)
        
        tau = self.TR/3
        
        #start  
        for i in range(self.ETL):
            self.op_T(self.RF1)
            self.op_Gx(tau, -1)
            self.op_Gx(tau, 1)
            self.t_echoPeak[i] = self.op_AD()
            self.op_Gx(tau, 1)

"""
Examples
"""
def CPMG2():
    #EPG_CPMGクラス使用
    cpmg = EPG_CPMG(ETL=100, FA_excite=90, FA_refocus=60, TE=20e-3)
    cpmg.run(t1=0.5, t2=0.1)
    cpmg.display_results()

def FISP2():
    #EPG_FISPクラス使用
    fisp = EPG_FISP(ETL=10, FA=30, TR=30e-3)
    fisp.run(t1=1, t2=0.1)
    fisp.display_results()

def CPMG():
    m0 = 1
    t1=0.5
    t2=0.1
    ETL=100
    FA_excite=90
    FA_refocus=60
    TE=20e-3
    epg = EPG(ETL*2+1)
    
    epg.setParams(m0, t1, t2)
    RF1 = epg.gen_Tmatrix(np.pi/2, FA_excite/180*np.pi)  #90y
    RF2 = epg.gen_Tmatrix(0, FA_refocus/180*np.pi)    #alpha_x
        
    epg.t_echoPeak = np.zeros((ETL), dtype = complex)
        
    #start
    epg.op_T(RF1)
    
    for i in range(ETL):
        epg.op_Gx(TE/2, 1)
        epg.op_T(RF2)
        epg.op_Gx(TE/2, 1)
        epg.t_echoPeak[i] = epg.op_AD()
    
    epg.display_results()

def FISP():
    m0 = 1
    t1=0.5
    t2=0.1
    ETL=100
    FA=30
    TR=30e-3
    
    epg = EPG(ETL+1)
    
    epg.setParams(m0, t1, t2)
    RF1 = epg.gen_Tmatrix(0, FA/180*np.pi)  #90y
    epg.t_echoPeak = np.zeros((ETL), dtype = complex)
        
    tau = TR/3
        
    #start  
    for i in range(ETL):
        epg.op_T(RF1)
        epg.op_Gx(tau, -1)
        epg.op_Gx(tau, 1)
        epg.t_echoPeak[i] = epg.op_AD()
        epg.op_Gx(tau, 1)
    
    epg.display_results()
    
if __name__ == '__main__': 
    CPMG()
    FISP()