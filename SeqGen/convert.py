# -*- coding: utf-8 -*-
"""
Created on Mon Mar 12 16:54:31 2018

@author: Yasuhiko Terada
"""

# -*- coding: utf-8 -*-
from PyQt5 import uic
     
fin = open('SeqGenMainwindow.ui', 'r')
fout = open('SeqGenMainwindow.py', 'w')
uic.compileUi(fin,fout,execute=False)
fin.close()
fout.close()