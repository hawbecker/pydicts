'''
  What it does: A dictionary of general tools that can be used for any data 
		
  Who made it: patrick.hawbecker@nrel.gov 
  When: 9/19/18
'''
import numpy as np
import matplotlib.pyplot as plt
from netCDF4 import Dataset as ncdf
from matplotlib import cm
import scipy.io.netcdf as netcdf
import matplotlib.colors as mcolors

def spectra1d(data,ds):
    ps = np.abs(np.fft.fft(data))**2
    freqs = np.fft.fftfreq(data.size, ds)
    return freqs,ps


def write1DVar(f,var,varstr,vartype,vardims,varunits):
    newvar        = f.createVariable(varstr,vartype,(vardims,))
    newvar[:]     = var
    newvar.units  = varunits

def calcTRI(zz,wndwx,wndwy):
    if (wndwx/2.0) - np.floor(wndwx/2.0) == 0.0:
        print 'wndwx must be odd...'
        return
    elif (wndwy/2.0) - np.floor(wndwy/2.0) == 0.0:
        print 'wndwy must be odd...'
        return
    else:
        Hwndwx = int(np.floor(wndwx/2))
        Hwndwy = int(np.floor(wndwy/2))
        nx = np.shape(zz)[1]; ny = np.shape(zz)[0]
        tri = np.zeros((ny,nx))
        for ii in range(Hwndwx+1,nx-Hwndwx-1):
            for jj in range(Hwndwy+1,ny-Hwndwy-1):
                tri[jj,ii] = np.sqrt(np.sum((zz[jj-Hwndwy:jj+Hwndwy+1,ii-Hwndwx:ii+Hwndwx+1] - zz[jj,ii])**2.0))
        return tri

    newvar.units  = varunits

def calcVRM(slope,aspect,wndw):
    if (wndw/2.0) - np.floor(wndw/2.0) == 0.0:
        print 'wndw must be odd...'
        return
    else:
        Hwndw = int(np.floor(wndw/2))
        nx = np.shape(slope)[1]; ny = np.shape(slope)[0]
        vrm = np.zeros((ny,nx))
        rugz   = np.cos(slope*np.pi/180.0)
        rugdxy = np.sin(slope*np.pi/180.0)
        rugx   = rugdxy*np.cos(aspect*np.pi/180.0)
        rugy   = rugdxy*np.sin(aspect*np.pi/180.0)

        for ii in range(Hwndw+1,nx-Hwndw-1):
            for jj in range(Hwndw+1,ny-Hwndw-1):
                vrm[jj,ii] = 1.0 - np.sqrt(\
                        np.sum(rugx[jj-Hwndw:jj+Hwndw+1,ii-Hwndw:ii+Hwndw+1])**2.0 + \
                        np.sum(rugy[jj-Hwndw:jj+Hwndw+1,ii-Hwndw:ii+Hwndw+1])**2.0 + \
                        np.sum(rugz[jj-Hwndw:jj+Hwndw+1,ii-Hwndw:ii+Hwndw+1])**2.0)/float(wndw**2)
        return vrm


# from: https://stackoverflow.com/questions/40929467/how-to-use-and-plot-only-a-part-of-a-colorbar-in-matplotlib
def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=-1):
    if n == -1:
        n = cmap.N
    new_cmap = mcolors.LinearSegmentedColormap.from_list(
        'trunc({name},{a:.2f},{b:.2f})'.format(name=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap
