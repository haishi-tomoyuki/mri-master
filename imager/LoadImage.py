# -*- coding: utf-8 -*-
#画像再構成プログラム
#Version 20
#

import numpy as np
from scipy import fftpack
import matplotlib.pyplot as plt
import os
import re
import time
from GIRF import Calc_Goutput


class Imager:
    """
    <Arguments> Filetype: 'img'->*.img, 'float32', ...
                nx=nx, ny=ny, nz=nz, fileorder=fileOrder, IsFFT=True, ZF=False, ZFx=1.0, ZFy=1.0, Zfz=1.0
    <Attribute> image : absolute image of reconstructed data
                image_p : phase image of recon. data
                image_c : recon. data (complex)
                nx, ny, nz : dimensions of images
                rawdata (complex) : k-space data
    Ver10以降, fft -> ifftに修正
    """
    def __init__(self, filename, filetype='img', nx=128, ny=128, nz=1, fileOrder='F', IsFFT = True, ZF = False, ZFx=1.0, ZFy=1.0, ZFz=1.0):
        self.filetype = filetype
        self.filename = filename
        self.nx = nx
        self.ny = ny
        self.nz = nz 
        
        if (filetype == 'img'):
            ###load&reconstruct
            self.loadImg()
                
            if (IsFFT == True):
                if self.datatype == 0:  #single slice, single echo
                    self.rawdata=np.reshape(np.array(self.rawdata), (self.nx,self.ny,self.nz), order='F')    
                    if (ZF==True):
                        self.zerofill(ZFx, ZFy, ZFz)
                    self.ifft()
                else:
                    if self.datatype == 1:  #multislice, single echo
                        self.nz = self.numSlices
                        self.rawdata = self.rawdata.reshape((self.nx, self.numSlices, self.ny), order='F')
                        self.rawdata = self.rawdata.transpose(0,2,1)
                        if (ZF==True):
                            self.zerofill(ZFx, ZFy, 1.0)
                        self.ifft_sliceBySlice()
                    if self.datatype == 2:  #single slice, multiecho
                        self.rawdata = self.rawdata.reshape((self.nx, self.numEchoes, self.ny, self.nz), order='F')
                        self.rawdata = self.rawdata.transpose(0,2,3,1)
                        if (ZF==True):
                            self.zerofill(ZFx, ZFy, ZFz)
                        self.ifft_multiEchoe()
                        
        else:
            ###load
            zdata_copy = np.fromfile(filename, dtype=filetype)    #ファイルから型を指定してロードし、zdata_copyに代入する（1次元配列として読み込まれる）
            self.image = np.reshape(zdata_copy, (nx, ny, nz), order=fileOrder)     #一次元配列→三次元配列へと変換        
    
    def loadImg(self):
        #ファイル読み込み
        header_size = int(348/4)
        """
        #read header
        f = np.fromfile(self.filename, dtype='short')        
        self.header = f[0:header_size]
        self.nx = int(self.header[21])
        self.ny = int(self.header[22])
        self.nz = int(self.header[23])
        """
        #read header2
        self.s_header = Standard_Header(filename=self.filename)
        self.nx = self.s_header.dim0
        self.ny = self.s_header.dim1
        self.nz = self.s_header.dim2        
        self.numSlices = self.s_header.dim4
        self.numEchoes = self.s_header.dim5
        
        if (self.numSlices == 1 and self.numEchoes == 1):  #single slice, single echo
            self.datatype = 0
        if (self.numSlices>1 and self.numEchoes == 1):  #multislice, single echo
            self.datatype = 1
        if (self.numSlices == 1 and self.numEchoes>1):  #single slice, multiecho
            self.datatype = 2

        #read data
        f = np.fromfile(self.filename, dtype='float32')        
        f = f[header_size:,]
        
        f = f.reshape(int(f.size/2), 2)
        raw_real = f[0:,0]
        raw_imag = f[0:,1]
        self.rawdata = np.vectorize(complex)(raw_real, raw_imag)    

    def zerofill(self, ZFx, ZFy, ZFz):  #sigleslice singleecho
        rawdata_copy = self.rawdata.copy()
        nx_org = self.rawdata.shape[0]
        ny_org = self.rawdata.shape[1]
        nz_org = self.rawdata.shape[2]
        nx_new = int(nx_org*ZFx)
        ny_new = int(ny_org*ZFy)
        nz_new = int(nz_org*ZFz)
        
        nx_s = int((nx_new-nx_org)/2)
        ny_s = int((ny_new-ny_org)/2)
        nz_s = int((nz_new-nz_org)/2)
        if self.datatype == 0 or self.datatype == 1:
            self.rawdata = np.zeros((nx_new, ny_new, nz_new), dtype=rawdata_copy.dtype)
            self.rawdata[nx_s:nx_s+nx_org, ny_s:ny_s+ny_org, nz_s:nz_s+nz_org] = rawdata_copy
        if self.datatype == 2:
            self.rawdata = np.zeros((nx_new, ny_new, nz_new, self.numEchoes), dtype=rawdata_copy.dtype)
            self.rawdata[nx_s:nx_s+nx_org, ny_s:ny_s+ny_org, nz_s:nz_s+nz_org,:] = rawdata_copy
 
        self.nx = nx_new
        self.ny = ny_new
        self.nz = nz_new
        #header情報も書き換え
        if (self.datatype==1):
            self.s_header.OverwriteHeaderDimInfo(nx_new, ny_new, 1, self.numSlices, self.numEchoes)
        else:
            self.s_header.OverwriteHeaderDimInfo(nx_new, ny_new, nz_new, self.numSlices, self.numEchoes)
 
        
    def ifft(self):
        #fft
        a = np.array([[[(-1)**(x+y+z) for z in range(self.nz)] for y in range(self.ny)] for x in range(self.nx)])
        self.image_c = fftpack.ifftn(self.rawdata*a, shape=(self.nx, self.ny, self.nz))
        self.image_c *= a
        self.image = np.abs(self.image_c)
        self.image_p = np.angle(self.image_c)
        
    def ifft_sliceBySlice(self):
        #2Dfft
        a = np.array([[(-1)**(x+y) for y in range(self.ny)] for x in range(self.nx)])
        self.image_c = np.zeros(self.rawdata.shape, dtype=self.rawdata.dtype)
        for k in range(self.nz):
            rawdata_2D = self.rawdata[:,:,k]
            image_c_2D = fftpack.ifftn(rawdata_2D*a, shape=(self.nx, self.ny))
            image_c_2D *= a
            self.image_c[:,:,k] = image_c_2D
        self.image = np.abs(self.image_c)
        self.image_p = np.angle(self.image_c)
        
    def ifft_multiEchoe(self):
        #echoごとにfft
        self.image_c = np.zeros(self.rawdata.shape, dtype=self.rawdata.dtype)
        a = np.array([[[(-1)**(x+y+z) for z in range(self.nz)] for y in range(self.ny)] for x in range(self.nx)])
        
        for nE in range(self.numEchoes):
            image_c_1 = fftpack.ifftn(self.rawdata[:,:,:,nE]*a, shape=(self.nx, self.ny, self.nz))
            image_c_1 *= a
            self.image_c[:,:,:,nE]=image_c_1

        self.image = np.abs(self.image_c)
        self.image_p = np.angle(self.image_c)
   
    def OverwriteHeaderDimInfo(self, nx, ny, nz, numSlices, numEchoes):
        self.s_header.OverwriteHeaderDimInfo(nx, ny, nz, numSlices, numEchoes)
        self.nx = self.s_header.dim0
        self.ny = self.s_header.dim1
        self.nz = self.s_header.dim2
        self.numSlices = self.s_header.dim4
        self.numEchoes = self.s_header.dim5
    
    def SaveImg(self, _filename):
        self.s_header.SaveImg(_filename, self.rawdata, datatype=self.datatype)

class Display3DImage:
    """
    <Arguments> image : 3D image, nx, ny, nz: dimensions of image
    """
    def __init__(self,image):
        self.image = image
        self.nx = np.shape(self.image)[0]
        self.ny = np.shape(self.image)[1]
        self.nz = np.shape(self.image)[2]
        self.slice_direction = 2
        self.slice_number = int(self.nz/2-1)       

        self.fig = plt.figure()
        self.slice_image = self.image[0:,0:,self.slice_number]
        plt.imshow(self.slice_image)
        plt.show()
      
        self.cid_p = self.fig.canvas.mpl_connect('scroll_event', self.on_press)
    
    def drawImage(self, slice_direction, slice_number):
        if (slice_direction == 0):   # yz plane
            self.slice_image = np.abs(self.image[slice_number,0:,0:])
        if (slice_direction == 1):   # xz plane
            self.slice_image = np.abs(self.image[0:,slice_number,0:])
        if (slice_direction == 2):   # xy plane
            self.slice_image = np.abs(self.image[0:,0:,slice_number])
        
        plt.imshow(self.slice_image)
        plt.show()
          
    def on_press(self,event):
        if (self.slice_direction == 0):
            max_slice_num = self.nx-1
        if (self.slice_direction == 1):
            max_slice_num = self.ny-1
        if (self.slice_direction == 2):
            max_slice_num = self.nz-1
     
        self.slice_number = min(max_slice_num, max(0, self.slice_number + event.step))
        self.drawImage(self.slice_direction, self.slice_number)
    
def SaveMhd(datafilename, data, order='F'):
    """
    dataをdatafilenameで保存
    """
    import os.path
    name, ext = os.path.splitext(datafilename)
    filename = name + '.mhd'
    
    try:
        f = open(filename, 'w')  
    except OSError as err:
        print("OS error: {0}".format(err))
    else:
        f.write('NDims = '+str(data.ndim) + '\n')
        f.write('DimSize =')
        for j in range(data.ndim):
            f.write(' '+str(data.shape[j]))
        f.write('\n')

        f.write('HeaderSize = -1\n')

        f.write('ElementSpacing =')
        for j in range(data.ndim):
            f.write(' 1')
        f.write('\n')

        f.write('Position =')
        for j in range(data.ndim):
            f.write(' 0')
        f.write('\n')
        """    
        if data.ndim == 3:
            f.write('NDims = 3\n')
            f.write('DimSize = '+str(nx) + ' ' + str(ny)+ ' ' + str(nz) +'\n')
            f.write('HeaderSize = -1\n')
            f.write('ElementSpacing = 1 1 1\n')
            f.write('Position = 0 0 0\n')
        """
        
        f.write('ElementByteOrderMSB = False\n')
        if (data.dtype == 'float32'):
            f.write('ElementType = MET_FLOAT\n')
        if (data.dtype == 'float64'):
            f.write('ElementType = MET_DOUBLE\n')
        if (data.dtype == 'uint16'):
            f.write('ElementType = MET_USHORT\n')
        if (data.dtype == 'int16'):
            f.write('ElementType = MET_SHORT\n')
        if (data.dtype == 'uint8'):
            f.write('ElementType = MET_UCHAR\n')
        if (data.dtype == 'int8'):
            f.write('ElementType = MET_CHAR\n')
        f.write('ElementDataFile = '+datafilename+'\n')
        f.close()
        
    if order == 'F':
        data.transpose().tofile(datafilename)  #Fortran orderで保存
    else:
        data.tofile(datafilename)  #Fortran orderで保存
  
    
class SeqInfo:
    """
    seqファイルから情報を抽出するクラス
    ---- Usage ----
    seq = SeqInfo(filename_seq)
    seq = SeqInfo(filename_seq, isSeqchart = False, GrampX=0, GrampY=0, GrampZ=0)
    ---- Attributes ----
    string: filename_seq
    array : DW, NR, N1, N2, ...
    array<string> : notes
    array<string> : comments
    int : actual NR, N_cut, N_ks
    string : filename_kxloc, filename_kyloc, filename_kzloc
    list<string>: event_list, event_GX, event_GY, event_GZ, event_RF, event_AD
    
    <<<generated when isSeqchart = True>>>
    array: seq_GX, seq_GY, seq_GZ, seq_RFx, seq_RFy, seq_AD
    """
    def __init__(self, filename_seq, isSeqchart = False, GrampX=0, GrampY=0, GrampZ=0):
        self.filename_seq = filename_seq
        try:
            f = open(filename_seq)  
        except OSError as err:
            print("OS error: {0}".format(err))
        else:
            self.notes = f.readlines() # 1行毎にファイル終端まで全て読む(改行文字も含まれる)
            f.close()
            self.Readheader()
            self.ReadLocationFiles()
        if isSeqchart:
            self.Seqchart(GrampX=GrampX, GrampY=GrampY, GrampZ=GrampZ)
            
        
    def ReadLocationFiles(self):
        self.actualNR = self.NR  #default
        self.n_cut = 0           #default
        
        for line in self.notes:
            keyword = ';number of k-sampling points='
            temp = line.find(keyword)
            if (temp>=0):
                self.n_ks = int(line[temp+len(keyword):-1])
            keyword = ';filename of kx_loc (double)='
            temp = line.find(keyword)
            if (temp>=0):
                self.filename_kxloc = line[temp+len(keyword):-1]
            keyword = ';filename of ky_loc (double)='
            temp = line.find(keyword)
            if (temp>=0):
                self.filename_kyloc = line[temp+len(keyword):-1]
            keyword = ';filename of kz_loc (double)='
            temp = line.find(keyword)
            if (temp>=0):
                self.filename_kzloc = line[temp+len(keyword):-1]
            keyword = ';matrix size of reconstructed images='
            temp = line.find(keyword)
            if (temp>=0):
                matrixSizeOfReconImages = line[temp+len(keyword):-1]   
                self.nx_recon = int(matrixSizeOfReconImages.split(',')[0])
                self.ny_recon = int(matrixSizeOfReconImages.split(',')[1])
                self.nz_recon = int(matrixSizeOfReconImages.split(',')[2])
            keyword = ';actualNR='
            temp = line.find(keyword)
            if (temp>=0):
                self.actualNR = int(line[temp+len(keyword):-1])
            keyword = ';n_cut='
            temp = line.find(keyword)
            if (temp>=0):
                self.n_cut = int(line[temp+len(keyword):-1])
                                    
    def Readheader(self):
        self.comments = []
        for line in self.notes:
            keyword = ':DW '
            temp = line.find(keyword)
            if (temp>=0):
                self.DW = int(line[temp+len(keyword):-1])
            keyword = ':NR '
            temp = line.find(keyword)
            if (temp>=0):
                self.NR = int(line[temp+len(keyword):-1])
            keyword = ':N1 '
            temp = line.find(keyword)
            if (temp>=0):
                self.N1 = int(line[temp+len(keyword):-1])
            keyword = ':N2 '
            temp = line.find(keyword)
            if (temp>=0):
                self.N2 = int(line[temp+len(keyword):-1])
            keyword = ':TR '
            temp = line.find(keyword)
            if (temp>=0):
                self.TR = int(line[temp+len(keyword):-1])
            keyword = ':SW '
            temp = line.find(keyword)
            if (temp>=0):
                self.SW = float(line[temp+len(keyword):-1])
            keyword = ':OF '
            temp = line.find(keyword)
            if (temp>=0):
                self.OF = float(line[temp+len(keyword):-1])
            keyword = ':EL '
            temp = line.find(keyword)
            if (temp>=0):
                self.EL = line[temp+len(keyword):-1]
            keyword = ':S1 '
            temp = line.find(keyword)
            if (temp>=0):
                self.S1 = int(line[temp+len(keyword):-1])
            keyword = ':S2 '
            temp = line.find(keyword)
            if (temp>=0):
                self.S2 = int(line[temp+len(keyword):-1])
            keyword = ';'
            temp = line.find(keyword)
            if (temp>=0):
                self.comments.append(line)

    def Eventlist(self):
        self.event_list = []
        for line in self.notes:
            seqevent = Seqevent(line, self.N1, self.N2, self.S1, self.S2)
            if seqevent.isTimeEvent == True:
                self.event_list.append(seqevent)
        self.Event_sort()
        
    def Event_sort(self):
        """
        イベント時間に従って並べ替え
        """
        for i in range(len(self.event_list)):
            min_pos = i
            for j in range(i+1, len(self.event_list)):
                if self.event_list[j].time < self.event_list[min_pos].time:
                    min_pos = j
            temp = self.event_list[min_pos]
            self.event_list[min_pos] = self.event_list[i]
            self.event_list[i] = temp
        
    
    def Seqchart(self, GrampX=0, GrampY=0, GrampZ=0):
        self.Eventlist()
        self.event_GX = []
        self.event_GY = []
        self.event_GZ = []
        self.event_RF = []
        self.event_AD = []
        for i in range(len(self.event_list)):
            type = self.event_list[i].type
            
            if type == "GX":
                self.event_GX.append(self.event_list[i])
            elif type == "GY":
                self.event_GY.append(self.event_list[i])    
            elif type == "GZ":
                self.event_GZ.append(self.event_list[i])
            elif type == "RF" or type == "PH":
                self.event_RF.append(self.event_list[i])
            elif type == "AD":
                self.event_AD.append(self.event_list[i])
            else:
                break
        t1 = time.time()
        #self.seq_GX = Calc_seqchart.Grad(self.event_GX, self.TR, self.N1, self.N2, Gramp=300)
        #self.seq_GY = Calc_seqchart.Grad(self.event_GY, self.TR, self.N1, self.N2, Gramp=300)
        #self.seq_GZ = Calc_seqchart.Grad(self.event_GZ, self.TR, self.N1, self.N2, Gramp=300)
        self.seq_GX = Calc_seqchart.Grad_simple(self.event_GX, self.TR, self.N1, self.N2, Gramp=GrampX)
        self.seq_GY = Calc_seqchart.Grad_simple(self.event_GY, self.TR, self.N1, self.N2, Gramp=GrampY)
        self.seq_GZ = Calc_seqchart.Grad_simple(self.event_GZ, self.TR, self.N1, self.N2, Gramp=GrampZ)
        self.seq_RFx, self.seq_RFy = Calc_seqchart.RF(self.event_RF, self.TR)
        self.seq_AD = Calc_seqchart.AD(self.event_AD, self.TR, self.NR, self.DW)
        t2 = time.time()
        print(t2-t1)
        

class Calc_seqchart:
    @staticmethod  #static method (インスタンス化しないで使えるスタティックメソッド)
    def Grad(event, TR, N1, N2, Gramp=0):
        """
        event_listからG(t)を作成する
        Gramp [us] : 傾斜制御
        戻り値: seqchart = np.array((TR_us, N1, N2))
        GIRFもこれを改造する?
        """
        TR_us = int(TR*1e3)
        seqchart = np.ones((TR_us, N1, N2)) * int('8000',16)  
        
        if Gramp==0:  #傾斜制御なし
            for i in range(len(event)):
               if any(event[i].table) == False:
                   seqchart[event[i].time:,:,:] = int(event[i].value,16)
               else:
                   if re.findall('<-.*?5', event[i].option):
                       for j in range(N1):
                           seqchart[event[i].time:,j,:] = int(event[i].table[j],16) 
                   if re.findall('<-.*?6', event[i].option):
                       for k in range(N2):
                           seqchart[event[i].time:,:,k] = int(event[i].table[k],16) 
        else:  #傾斜制御あり
            for i in range(len(event)):
               if any(event[i].table) == False:         
                   for j in range(N1):
                       for k in range(N2):
                           val_before = seqchart[event[i].time, j, k]
                           val_after = int(event[i].value,16)
                           slope = (val_after - val_before)/Gramp
                           for ni in range(Gramp):
                               seqchart[event[i].time + ni, j, k] = slope*ni + val_before
                           seqchart[event[i].time+Gramp:, j, k] = val_after
                       
               else:
                   if re.findall('<-.*?5', event[i].option):
                       for j in range(N1):
                           for k in range(N2):
                               val_before = seqchart[event[i].time, j, k]
                               val_after = int(event[i].table[j],16)
                               slope = (val_after - val_before)/Gramp
                               for ni in range(Gramp):
                                   seqchart[event[i].time + ni, j, k] = slope*ni + val_before
                               seqchart[event[i].time+Gramp: ,j ,k] = val_after 

                   if re.findall('<-.*?6', event[i].option):
                       for j in range(N1):
                           for k in range(N2):
                               val_before = seqchart[event[i].time, j, k]
                               val_after = int(event[i].table[k],16)
                               slope = (val_after - val_before)/Gramp
                               for ni in range(Gramp):
                                   seqchart[event[i].time + ni, j, k] = slope*ni + val_before
                               seqchart[event[i].time+Gramp:,j ,k] = val_after 
            
        return seqchart
    
    @staticmethod
    def Grad_simple(event, TR, N1, N2, Gramp=0):
        """
        event_listからG(t)を作成する
        シンプルバージョン
        Gramp [us] : 傾斜制御
        戻り値: seqchart = np.array((TR_us, N1))
        """
        TR_us = int(TR*1e3)
        Ny = max(N1, N2)
        seqchart = np.ones((TR_us, Ny)) * int('8000',16)
        
        if Gramp==0: #傾斜制御なし
            for i in range(len(event)):
               if any(event[i].table) == False:
                   seqchart[event[i].time:,:] = int(event[i].value,16)
               else:
                   if re.findall('<-.*?5', event[i].option):
                       for j in range(N1):
                           seqchart[event[i].time:,j] = int(event[i].table[j],16) 
                   if re.findall('<-.*?6', event[i].option):
                       for k in range(N2):
                           seqchart[event[i].time:,k] = int(event[i].table[k],16) 
        else: #傾斜制御あり
            for i in range(len(event)):
               if any(event[i].table) == False:         
                   for j in range(Ny):
                       val_before = seqchart[event[i].time,j]
                       val_after = int(event[i].value,16)
                       slope = (val_after - val_before)/Gramp
                       for ni in range(Gramp):
                           seqchart[event[i].time + ni, j] = slope*ni + val_before
                       seqchart[event[i].time+Gramp:, j] = val_after
                       
               else:
                   if re.findall('<-.*?5', event[i].option):
                       for j in range(N1):
                           val_before = seqchart[event[i].time,j]
                           val_after = int(event[i].table[j],16)
                           slope = (val_after - val_before)/Gramp
                           for ni in range(Gramp):
                               seqchart[event[i].time + ni,j] = slope*ni + val_before
                           seqchart[event[i].time+Gramp:,j] = val_after 

                   if re.findall('<-.*?6', event[i].option):
                       for k in range(N2):
                           val_before = seqchart[event[i].time,k]
                           val_after = int(event[i].table[k],16)
                           slope = (val_after - val_before)/Gramp
                           for ni in range(Gramp):
                               seqchart[event[i].time + ni,k] = slope*ni + val_before
                           seqchart[event[i].time+Gramp:,k] = val_after 
        return seqchart
    
    @staticmethod
    def AD(event, TR, NR, DW):
        TR_us = int(TR*1e3)
        seqchart = np.zeros((TR_us))
        AD_time = NR*DW
        for i in range(len(event)):
            if event[i].type == 'AD':
                seqchart[event[i].time : event[i].time+AD_time] = 1
        return seqchart

    @staticmethod
    def RF(event, TR):
        """
        RFpulseShapeは000Fの場合にユーザー定義パルスを入れて使う
        """
        TR_us = int(TR*1e3)
        seqchart_RFx = np.zeros((TR_us))
        seqchart_RFy = np.zeros((TR_us))
        phase = 0  #RFpulseの位相
    
        for i in range(len(event)):
            if not hasattr(event[i], 'RFpulseShapeX'):                           
                RFpulseShapeX = np.ones((120))
                RFpulseShapeY = np.zeros((120))
                isUD = False
            else: 
                RFpulseShapeX = event[i].RFpulseShapeX
                RFpulseShapeY = event[i].RFpulseShapeY
                isUD = True
                
            if event[i].type == 'PH':
                phase = int(event[i].value,16) / 0x0100 * np.pi/2  #radian
            if event[i].type == 'RF':
                if event[i].value == '0002':  #90 hard pulse
                    seqchart_RFx[event[i].time : event[i].time+120] = 90*np.cos(phase)
                    seqchart_RFy[event[i].time : event[i].time+120] = 90*np.sin(phase)
                if event[i].value == '0003':  #180 hard pulse
                    seqchart_RFx[event[i].time : event[i].time+120] = 180*np.cos(phase)
                    seqchart_RFy[event[i].time : event[i].time+120] = 180*np.sin(phase)
                if isUD:
                    seqchart_RFx[event[i].time : event[i].time+len(RFpulseShapeX)] = RFpulseShapeX
                    seqchart_RFy[event[i].time : event[i].time+len(RFpulseShapeY)] = RFpulseShapeY
                else:
                    if event[i].value == '000D':  #multislice 90 pulse
                        seqchart_RFx[event[i].time : event[i].time+1000] = 90*np.cos(phase)
                        seqchart_RFy[event[i].time : event[i].time+1000] = 90*np.sin(phase)
                    if event[i].value == '000E':  #multislice 180 pulse
                        seqchart_RFx[event[i].time : event[i].time+1000] = 180*np.cos(phase)
                        seqchart_RFy[event[i].time : event[i].time+1000] = 180*np.sin(phase)
        return seqchart_RFx, seqchart_RFy
    

"""
seq = SeqInfo("00001002-2DRadial_GE_10cm_ny256-145548.seq", isSeqchart=True)
seq = SeqInfo("00001010-2DGE_projection_10x10cm-151701.seq", isSeqchart=True)
plt.plot(seq.seq_GX)

"""      

        
class Seqevent:
    """
    <attribute>
    time : [us]
    type : GX, GY, GZ, AD, RF
    value : 8000など
    option　： '<-v5{8000, ...}'
    table : [8000, C000, ...]
    isTimeEvent : True or False
    line : 00.001.000.0 GX 8000
    filename : 'c:\MRT\...'
    RFpulseShapeX, RFpulseShapeY (if any)
    """
    def __init__(self, line, N1, N2, S1, S2):
        self.line = line
        keyword = r"^\d\d.\d\d\d.\d\d\d.\d"
        temp = re.findall(keyword, line)
        if temp:
            #tempを秒（文字列）→（数字）にする
            self.time = int(float(line[0:2]+"."+line[3:6]+line[7:10]+line[11:12])*1e6)      #time [us]
            self.type = line[13:15]                                                         #'GX', ...
            self.value = line[16:20]                                                        #8000
            self.option = line[20:]                                                       #<-v5{8000, ...}
            self.table = self.Get_Tablelist(self.option, N1, N2, S1, S2)
            self.isTimeEvent = True
            
            #ファイル名があれば取り出す
            temp2=re.findall("=", self.option)
            if temp2:
                self.filename =self.option[2:len(self.option)-1]
                #RFファイルを読む
                if self.type == 'RF':
                    self.Get_RFPulseFromFile(self.filename)
                
            
        else:
            self.isTimeEvent = False
            
    def Get_Tablelist(self, option, N1, N2, S1, S2):
        #encode tableの値を解釈
        table = []
        temp = re.findall("<-v", option)
        if temp:
            table = option.split(',')
            table[0] = table[0][5:]                         #<-v5を取り除く
            table[len(table)-1] = table[len(table)-1][:4]   #}\nを取り除く        
    
        temp = re.findall("<-e5", option)
        if temp:
            for i in range(N1):
                table.append(hex(int(32768 - N1*S1/2 + i*S1)))

        temp = re.findall("<-e6", option)
        if temp:
            for i in range(N2):
                table.append(hex(int(32768 - N2*S2/2 + i*S2)))

        temp = re.findall("<-c5", option)
        if temp:
            for i in range(N1):
                table.append(hex(int(32768 + N1*S1/2 - i*S1)))

        temp = re.findall("<-c6", option)
        if temp:
            for i in range(N2):
                table.append(hex(int(32768 + N2*S1/2 - i*S2)))

        return table
    
    def Get_TablelistFromFile(self, filename, N1, N2, S1, S2):
        """
        ファイルからGradientテーブル情報を得る(未実装)
        self.tableに格納
        """
        pass
    
    def Get_RFPulseFromFile(self, filename):
        """
        ファイルからRFpulse情報を得る
        self.RFpulseShapeX
        self.RFpulseShapeY
        を作る
        """
        if filename != '':
            try:
                f = open(filename)  
            except OSError as err:
                print("OS error: {0}".format(err))
            else:
                lines = f.readlines() # 1行毎にファイル終端まで全て読む(改行文字も含まれる)
                f.close()
                
                temp = lines[1].split('\t')
                Npoints = int(lines[0])
                ReadoutDwell =  int(int(temp[0]) *0.1)                          #us
                duration = Npoints * ReadoutDwell                #duration[us]
                amp = float(temp[1])  #増幅率
                BW = float(temp[2])*1000                           #BW[Hz]
                RFpulseShapeX = np.ones((duration))  #1us刻み
                RFpulseShapeY = np.zeros((duration))  #1us刻み
                for i in range(len(lines)-2):
                    temp = lines[i+2].split('\t')
                    RFpulseShapeX[i*ReadoutDwell:(i+1)*ReadoutDwell] = int(temp[0])*amp
                    RFpulseShapeY[i*ReadoutDwell:(i+1)*ReadoutDwell] = int(temp[1])*amp
                    
        self.RFpulseShapeX = RFpulseShapeX
        self.RFpulseShapeY = RFpulseShapeY


class Calc_kloc:
    """
        
    """
    @staticmethod
    def Calc_kloc_nominal(seq_G, seq_AD, I_max = 10, G_eff = 0.272, DW=30, actualNR=256, OS = 1):
        """
        seq_G:seq_GX
        AD_event:event_AD
        AD_time:NR*DW [us]
        I_max:[A]
        G_eff:[G/cm/A]
        
        return
        kloc_DW:[1/m]
        """
        gradient_I = I_max/32768*(seq_G-32768)
        gradient = gradient_I* G_eff * 4.2577e3  *1e2
 
        for i in range(len(seq_AD)):
            if seq_AD[i] == 1:
                AD_start = i
                break
        for i in range(AD_start+1,len(seq_AD)):
            if seq_AD[i] == 0:
                AD_end = i
                break
        AD_time = AD_end - AD_start
        gradient_cut = gradient[AD_start:AD_end,]
        #k_loc計算
        kloc_us = np.zeros(np.shape(gradient_cut))
        for i in range(len(kloc_us)):
            kloc_us[i,:] = gradient_cut[i,:]+kloc_us[i-1,:]   #積分
        kloc_DW = np.zeros((actualNR,gradient_cut.shape[1]))
        for i in range(actualNR):
            kloc_DW[i,:] = kloc_us[int(i*DW/OS),:]        
        kloc_DW = kloc_DW.reshape(kloc_DW.size, order='F')
        return kloc_DW*1e-6
        
        
    @staticmethod
    def Calc_gradient_predict(filename_GIRF,filename_GIRFx, seq_G, f_cut=20, amp_cut=1.2, offset=32768):
        """
        GIRF予測のgradientを計算
        """
        GIRF = np.fromfile(filename_GIRF, dtype = "complex128")
        GIRF_x = np.fromfile(filename_GIRFx,dtype="float64")
        gradient_predict = np.zeros(np.shape(seq_G))
        gradient_x = np.arange(seq_G.shape[0],dtype="float64")*1e-3 #[ms]
        for i in range(seq_G.shape[1]):
            print(i,end=",")
            gradient_predict[:,i]= Calc_Goutput(seq_G[:,i]-offset, gradient_x, GIRF, GIRF_x, f_cut=f_cut, amp_cut=amp_cut)
        return gradient_predict+offset
                    
        
                
        
        

class Standard_Header:
    def __init__(self, filename=''):
        self.filename = filename
        if (os.path.isfile(filename)):
            self.readHeader()
        else:
            sizeofheader = 348
            self.header = bytearray(np.zeros(sizeofheader,dtype='bytes').tobytes())
            self.header[0] = sizeofheader.to_bytes(2, 'little')[0]
            self.header[1] = sizeofheader.to_bytes(2, 'little')[1]
            self.header[48] = (1).to_bytes(2, 'little')[0]  #dim3=1
            self.header[49] = (1).to_bytes(2, 'little')[1]
            self.header[50] = (1).to_bytes(2, 'little')[0]  #dim4=1
            self.header[51] = (1).to_bytes(2, 'little')[1]
            self.header[52] = (1).to_bytes(2, 'little')[0]  #dim5=1
            self.header[53] = (1).to_bytes(2, 'little')[1]
            self.header[54] = (1).to_bytes(2, 'little')[0]  #dim6=1
            self.header[55] = (1).to_bytes(2, 'little')[1]
            
    def readHeader(self):
        infile = open(self.filename, 'rb')
        self.header = bytearray(infile.read(348))
        self.dim0 = int('0x'+"{0:0>2X}".format(self.header[43])+"{0:0>2X}".format(self.header[42]), 16)
        self.dim1 = int('0x'+"{0:0>2X}".format(self.header[45])+"{0:0>2X}".format(self.header[44]), 16)
        self.dim2 = int('0x'+"{0:0>2X}".format(self.header[47])+"{0:0>2X}".format(self.header[46]), 16)
        self.dim3 = int('0x'+"{0:0>2X}".format(self.header[49])+"{0:0>2X}".format(self.header[48]), 16)
        self.dim4 = int('0x'+"{0:0>2X}".format(self.header[51])+"{0:0>2X}".format(self.header[50]), 16)
        self.dim5 = int('0x'+"{0:0>2X}".format(self.header[53])+"{0:0>2X}".format(self.header[52]), 16)
        self.dim6 = int('0x'+"{0:0>2X}".format(self.header[55])+"{0:0>2X}".format(self.header[54]), 16)
        infile.close()
    
    def OverwriteHeaderDimInfo(self, nx, ny, nz, numSlices, numEchoes):
        self.header[42] = nx.to_bytes(2, 'little')[0]
        self.header[43] = nx.to_bytes(2, 'little')[1]
        self.header[44] = ny.to_bytes(2, 'little')[0]
        self.header[45] = ny.to_bytes(2, 'little')[1]
        self.header[46] = nz.to_bytes(2, 'little')[0]
        self.header[47] = nz.to_bytes(2, 'little')[1]
        self.header[50] = numSlices.to_bytes(2, 'little')[0]
        self.header[51] = numSlices.to_bytes(2, 'little')[1]
        self.header[52] = numEchoes.to_bytes(2, 'little')[0]
        self.header[53] = numEchoes.to_bytes(2, 'little')[1]
        self.dim0 = int('0x'+"{0:0>2X}".format(self.header[43])+"{0:0>2X}".format(self.header[42]), 16)
        self.dim1 = int('0x'+"{0:0>2X}".format(self.header[45])+"{0:0>2X}".format(self.header[44]), 16)
        self.dim2 = int('0x'+"{0:0>2X}".format(self.header[47])+"{0:0>2X}".format(self.header[46]), 16)            
        self.dim4 = int('0x'+"{0:0>2X}".format(self.header[51])+"{0:0>2X}".format(self.header[50]), 16)
        self.dim5 = int('0x'+"{0:0>2X}".format(self.header[53])+"{0:0>2X}".format(self.header[52]), 16)
               
    def SaveImg(self, _filename, rawdata_c, datatype=0):
        """
        _filename : filename of output file
        rawdata_c : raw k-space data [dim0, dim1, dim2, dim4 (dim5)]
        datatype : signle slice & echo (0), multislice & single echo (1), single slice & multi echo (2)
        """
        outfile = open(_filename, 'wb')
        outfile.write(self.header)
        if datatype == 0: #singleslice singleecho
            raw_1D = rawdata_c.reshape((self.dim0*self.dim1*self.dim2), order='F')
        if datatype == 1:  #multislice, single echo
            raw = rawdata_c.transpose(0,2,1)
            raw_1D = raw.reshape((self.dim0*self.dim1*self.dim4), order='F')
        if datatype == 2:  #singleslice, multiecho
            raw = rawdata_c.transpose(0,3,1,2)
            raw_1D = raw.reshape((self.dim0*self.dim1*self.dim2*self.dim5), order='F')

        for i in range(len(raw_1D)):
            outfile.write(raw_1D[i].astype('complex64'))
        outfile.close()
    
    def SaveImgFromImage(self, _filename, image_real, image_imag, datatype=0):
        """
        _filename : filename of output file
        image_real : real part of image
        image_imag : imaginary part of image
        datatype : signle slice & echo (0), multislice & single echo (1), single slice & multi echo (2)
        
        shape of image_real & image_imag
        datatype 0 -> [nx, ny, nz]
        datatype 1 -> [nx, ny, numOfSlices]
        datatype 2 -> [nx, ny, nz, numOfEchoes]
        """
        #nx, ny, nz, numOfSlices, numOfEchoesの決定
        nx = image_real.shape[0]
        ny = image_real.shape[1]
        if datatype == 0:
            nz = image_real.shape[2]
            numOfSlices = 1
            numOfEchoes = 1
        if datatype == 1:
            nz = 1
            numOfSlices = image_real.shape[2]
            numOfEchoes = 1
        if datatype == 2:
            nz = image_real.shape[2]
            numOfSlices = 1
            numOfEchoes = image_real.shape[3]

        self.OverwriteHeaderDimInfo(nx, ny, nz, numOfSlices, numOfEchoes)
        image_c = np.vectorize(complex)(image_real, image_imag)
        
        if datatype == 0: #singleslice singleecho
            #fft
            a = np.array([[[(-1)**(x+y+z) for z in range(nz)] for y in range(ny)] for x in range(nx)])
            rawdata_c = fftpack.fftn(image_c*a, shape=(nx, ny, nz))
            rawdata_c *= a
            raw_1D = rawdata_c.reshape((nx*ny*nz), order='F')
        if datatype == 1:  #multislice, single echo
            #fft
            rawdata_c = image_c.copy()
            a = np.array([[(-1)**(x+y) for y in range(ny)] for x in range(nx)])
            for k in range(numOfSlices):
                image_2D_c = image_c[:,:,k]
                rawdata_2D_c = fftpack.fftn(image_2D_c*a, shape=(nx, ny))
                rawdata_2D_c *= a
                rawdata_c[:,:,k] = rawdata_2D_c                
            raw = rawdata_c.transpose(0,2,1)
            raw_1D = raw.reshape((nx*ny*numOfSlices), order='F')
        if datatype == 2:  #singleslice, multiecho
            rawdata_c = image_c.copy()
            for nE in range(numOfEchoes):             
                image_3D_c = image_c[:,:,:,nE]
                a = np.array([[[(-1)**(x+y+z) for z in range(nz)] for y in range(ny)] for x in range(nx)])
                rawdata_3D_c = fftpack.fftn(image_3D_c*a, shape=(nx, ny, nz))
                rawdata_3D_c *= a
                rawdata_c[:,:,:,nE] = rawdata_3D_c
            raw = rawdata_c.transpose(0,3,1,2)
            raw_1D = raw.reshape((nx*ny*nz*numOfEchoes), order='F')
            
        #save
        outfile = open(_filename, 'wb')
        outfile.write(self.header)
        for i in range(len(raw_1D)):
            outfile.write(raw_1D[i].astype('complex64'))
        outfile.close()


def test():
    #load seq with seqchart
    seq = SeqInfo('00001002-2DRadial_GE_10cm_ny256-145548.seq', isSeqchart=True)
    plt.plot(seq.seq_GX)
    #load img
  #  img = Imager('test.img')
    
if __name__ == '__main__':
    test()
        
