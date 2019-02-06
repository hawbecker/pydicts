'''
  What it does: This dictionary contains functions for reading
                observational data.

  Who made it: patrick.hawbecker@nrel.gov
  When: 9/02/18
'''
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm    
from netCDF4 import Dataset as ncdf
import pickle
import subprocess
import pandas as pd

class m2data():
    def __init__(self,fd):
        self.fdir = fd
        self.getvarns()
        self.getdata()

    def getvarns(self):
        f = open(self.fdir,'r')
        self.varns = f.readline().replace('\n','').split(',')
        self.nt = sum(1 for line in f)
        f.close()

    def getdata(self):
        f = pd.read_csv(self.fdir)
        gotTemp = False
        gotWind = False
        gotWdir = False
        gotTI   = False
        for var in self.varns:
            vind = self.varns.index(var)
            dat  = f[var]
            if 'DATE' in var:
                self.obdate = dat
            elif 'MST' in var:
                self.obtime = np.zeros((self.nt))
                tt = 0
                for time in dat:
                    hours = float(time.split(':')[0])
                    mins  = float(time.split(':')[1])
                    self.obtime[tt] = (hours + mins/60.0)/24.0
                    tt += 1
            elif 'Global PSP' in var:
                self.radiation = dat
            elif 'Temperature @' in var:
                if gotTemp == False:
                    ntemps = 0
                    tempind = 0
                    for vv in self.varns: 
                        if 'Temperature @' in vv: ntemps += 1
                    self.ntemp = ntemps
                    self.temp  = np.zeros((self.nt,ntemps))
                    self.tempZ = np.zeros((ntemps))
                    gotTemp = True
                self.temp[:,tempind] = dat
                self.tempZ[tempind]  = float(''.join(i for i in var if i.isdigit()))
                tempind += 1
            elif 'Station Pressure' in var:
                self.pressure = dat
            elif 'Avg Wind Speed @' in var:
                if gotWind == False:
                    nwinds = 0
                    windind = 0
                    for vv in self.varns: 
                        if 'Avg Wind Speed @' in vv: nwinds += 1
                    self.nwind = nwinds
                    self.wspd  = np.zeros((self.nt,nwinds))
                    self.windZ = np.zeros((nwinds))
                    gotWind = True
                self.wspd[:,windind] = dat
                self.windZ[windind]  = float(''.join(i for i in var if i.isdigit()))
                windind += 1
            elif 'Avg Wind Direction @' in var:
                if gotWdir == False:
                    windind = 0
                    self.wdir  = np.zeros((self.nt,nwinds))
                    gotWdir = True
                self.wdir[:,windind] = dat
                windind += 1
            elif 'Turbulence Intensity @' in var:
                if gotTI == False:
                    nti   = 0
                    tiind = 0
                    for vv in self.varns: 
                        if 'Turbulence Intensity @' in vv: nti += 1
                    self.nTI = nti
                    self.TI  = np.zeros((self.nt,nti))
                    self.TIZ = np.zeros((nti))
                    gotTI = True
                self.TI[:,tiind] = dat
                self.TIZ[tiind] = float(''.join(i for i in var if i.isdigit()))
                tiind += 1
            elif 'Friction Velocity ' in var:
                self.ustar = dat
            elif 'Est Surface Roughness ' in var:
                self.z0 = dat
            elif 'u* Quality Control ' in var:
                self.ustarQC = dat
