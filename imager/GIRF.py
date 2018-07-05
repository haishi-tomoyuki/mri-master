# -*- coding: utf-8 -*-
"""
Created on Thu Oct 19 17:56:37 2017

@author: nakao
"""
import numpy as np
from scipy.fftpack import fft, fftfreq, fftshift


def Calc_Goutput(G_in, G_input_x,GIRF,GIRF_x, f_cut, amp_cut):
    """
    入力波形をGIRFで歪ませて出力したものを戻す
    G_in: 入力波形(1us刻み,時間領域) [1D real]
    G_input_x: 入力波形の時間軸[ms] [1D real]
    GIRF: GIRF関数(周波数領域) [1D complex]
    GIRF_x: GIRF関数の周波数軸[kHz] [1D real]
    """
    G_input_long, G_input_x_long,GIRF_1us,GIRF_1us_x = GIRF_Interpolation(G_in, G_input_x, GIRF, GIRF_x)
    G_predict = GIRF_Predict(G_input_long,GIRF_1us, GIRF_1us_x,f_cut,amp_cut)
    G_predict_cut = G_predict[:len(G_in)]
    return G_predict_cut


def GIRF_Interpolation(G_in,G_input_x, GIRF, GIRF_x):
    """
    G_in：入力関数のyデータ　[1us刻み]
    G_input_x：入力関数のx軸(ms)
    GIRF：伝達関数のyデータ(complex)
    GIRF_x：周波数軸(kHz)
    """
    """
    G_INを1us刻みでGIRFに合わせて延長
    周波数分解能を逆数をとる
    """
    """
    戻り値：
    G_input_long: 入力関数をGIRF関数の逆数の時間Tまで延長したもの
    G_input_x_long:　↑の時間軸(x軸) [ms]
    GIRF_interp:  GIRF関数の周波数軸を、入力関数の周波数軸に合わせてinterpolateしたもの
    fG_input_x: ↑の周波数軸(x軸) [kHz] 
    """
    
    T_GIRF = 1/(GIRF_x[1]-GIRF_x[0])#[ms]
   # N_long = int(T/1e-3)
    T_G_input = G_input_x[len(G_input_x)-1]
    if T_GIRF >= T_G_input:
        G_input_x_long = np.arange(0,T_GIRF,1e-3)
        G_input_long = np.zeros(len(G_input_x_long))
        G_input_long[0:len(G_in)] = G_in
    else:
        G_input_long = G_in
        G_input_x_long = G_input_x
    
    N = len(G_input_x_long)
    dt = G_input_x_long[1]-G_input_x_long[0]
    #入力関数のフーリエ変換の周波数軸[kHz]
    
    fG_input_x = np.fft.fftshift(np.fft.fftfreq(N,dt))

    GIRF_real = np.real(GIRF)
    GIRF_imag = np.imag(GIRF)
    GIRF_interp_real = np.interp(fG_input_x, GIRF_x, GIRF_real)
    GIRF_interp_imag = np.interp(fG_input_x, GIRF_x, GIRF_imag)
    GIRF_interp = np.vectorize(complex)(GIRF_interp_real, GIRF_interp_imag) 
    #GIRF_xの範囲外はゼロにする
    GIRF_interp = np.where(abs(fG_input_x) > GIRF_x.max(), 0, GIRF_interp)
 
    return G_input_long, G_input_x_long, GIRF_interp,fG_input_x    
    

def GIRF_Predict(G_input_long, GIRF_1us, GIRF_1us_x, f_cut, amp_cut):#G_inはG_input[:,1]
    temp = np.zeros(len(G_input_long))
    G_in_temp = np.r_[temp,G_input_long]
    G_input_real  = G_in_temp
    G_input_imag = np.zeros(G_input_real.shape)
    G_input_c = np.vectorize(complex)(G_input_real,G_input_imag)
    
    
    a = np.array([(-1)**x for x in range(len(G_input_real))])
    
    fG_input = np.fft.fft(G_input_c*a)
    fG_input *= a

    #lowpass: GIRF->GIRF_LPF
    gaussian = np.exp(-GIRF_1us_x**2/f_cut**2)
    
    """
    for i in range(len(GIRF)):
        if amp_cut < np.abs(GIRF[i]):# and np.abs(GIRF_x[i])>f_cut:
            GIRF[i] = 0            
    """
    GIRF_1us = np.where(amp_cut < np.abs(GIRF_1us), 0, GIRF_1us)
    
    GIRF_LPF = gaussian * GIRF_1us
    #GIRF_LPF = np.where(amp_cut < np.abs(GIRF_LPF), 0, GIRF_LPF)

    GIRF_LPF_real = np.real(GIRF_LPF)
    GIRF_LPF_imag = np.imag(GIRF_LPF)
    
    xp = np.arange(0,len(GIRF_1us),1)
    x = np.arange(0, len(GIRF_1us), 0.5)
    
    GIRF_LPF_interp = np.vectorize(complex)(np.interp(x,xp,GIRF_LPF_real),np.interp(x,xp,GIRF_LPF_imag))
    
    fG_output_predict = GIRF_LPF_interp * fG_input

    
    G_output_predict  = np.fft.ifft(fG_output_predict*a)*a
    G_predict = np.real(G_output_predict[len(G_input_long):])
    
    #plt.plot(abs(GIRF_LPF))
    return G_predict