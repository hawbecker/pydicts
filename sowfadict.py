'''
  What it does: This dictionary contains functions for reading
                SOWFA data

  Who made it: patrick.hawbecker@nrel.gov
  When: 5/15/18
'''
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm    
from netCDF4 import Dataset as ncdf
import pickle
import subprocess
import os
import wrfdict as wrfdict

class probe():
    def __init__(self,fd,varn):
        self.fdir = fd
        self.getxyz(varn[0])
        for var in varn:
            self.getdata(var)

    def getxyz(self,var):
        z = []; nprbs = 0
        print self.fdir
        f = open('%s%s' % (self.fdir,var),'r')
        for line in f:
            if line[0] == '#':
                if 'Probe' in line and '(' in line:
                    nprbs += 1
                    self.x = line.replace('(',' ').replace(')',' ').split()[-3]
                    self.y = line.replace('(',' ').replace(')',' ').split()[-3]
                    z.append(float(line.replace('(',' ').replace(')',' ').split()[-1]))
        self.z = z

    def getdata(self,var):
        nlines = sum(1 for line in open('%s%s' % (self.fdir,var),'r'))
        f = open('%s%s' % (self.fdir,var),'r')
        tt = 0; header = 0; z = []; nt = 0; init = False
        for line in f:
            if line[0] == '#':
                header += 1
            else:
                nt = header-2
                if init == False:
                    time = np.zeros((nlines-header))
                    if var == 'U':
                        u = np.zeros((nt,nlines-header))
                        v = np.zeros((nt,nlines-header))
                        w = np.zeros((nt,nlines-header))
                    elif var == 'T':
                        T = np.zeros((nt,nlines-header))
                    init = True
                line = line.replace('(',' ').replace(')',' ').split()
                time[tt] = line[0]
                if var == 'U':
                    u[:,tt] = line[1::3]
                    v[:,tt] = line[2::3]
                    w[:,tt] = line[3::3]
                elif var == 'T':
                    T[:,tt] = line[1:]
                tt += 1
        f.close()
        self.time = time
        if var == 'U':
            self.u = u
            self.v = v
            self.w = w
        elif var == 'T':
            self.T = T


class probeLine():
    def __init__(self,fd,lineaxis,varn):
        self.fdir = fd
        self.lineAxis = lineaxis
        self.getxyz(varn[0])
        for var in varn:
            self.getdata(var)

    def getxyz(self,var):
        x = []; y = []
        z = []; nprbs = 0
        f = open('%s%s' % (self.fdir,var),'r')
        for line in f:
            if line[0] == '#':
                if 'Probe' in line and '(' in line:
                    nprbs += 1
                    x.append(float(line.replace('(',' ').replace(')',' ').split()[-3]))
                    y.append(float(line.replace('(',' ').replace(')',' ').split()[-2]))
                    z.append(float(line.replace('(',' ').replace(')',' ').split()[-1]))
        f.close()
        if self.lineAxis == 'y':
            self.towerpos = np.unique(y)
            self.ntowers  = np.shape(self.towerpos)[0]
        elif self.lineAxis == 'x':
            self.towerpos = np.unique(x)
            self.ntowers  = np.shape(self.towerpos)[0]
        self.nlevels = nprbs/self.ntowers

        tx = np.zeros((self.ntowers,self.nlevels))
        ty = np.zeros((self.ntowers,self.nlevels))
        tz = np.zeros((self.ntowers,self.nlevels))
        if self.lineAxis == 'y':
            for tt in np.arange(0,self.ntowers):
                tx[tt,:] = x[np.where(y==self.towerpos[tt])[0][0]:np.where(y==self.towerpos[tt])[0][-1]+1]
                ty[tt,:] = y[np.where(y==self.towerpos[tt])[0][0]:np.where(y==self.towerpos[tt])[0][-1]+1]
                tz[tt,:] = z[np.where(y==self.towerpos[tt])[0][0]:np.where(y==self.towerpos[tt])[0][-1]+1]
        elif self.lineAxis == 'x':
            for tt in np.arange(0,self.ntowers):
                tx[tt,:] = x[np.where(x==self.towerpos[tt])[0][0]:np.where(x==self.towerpos[tt])[0][-1]+1]
                ty[tt,:] = y[np.where(x==self.towerpos[tt])[0][0]:np.where(x==self.towerpos[tt])[0][-1]+1]
                tz[tt,:] = z[np.where(x==self.towerpos[tt])[0][0]:np.where(x==self.towerpos[tt])[0][-1]+1]
        self.x = tx
        self.y = ty
        self.z = tz

    def getdata(self,var):
        nlines = sum(1 for line in open('%s%s' % (self.fdir,var),'r'))
        f = open('%s%s' % (self.fdir,var),'r')
        tt = 0; header = 0; z = []; init = False
        for line in f:
            if line[0] == '#':
                header += 1
            else:
                if init == False:
                    time = np.zeros((nlines-header))
                    if var == 'U':
                        u = np.zeros((self.ntowers,self.nlevels,nlines-header))
                        v = np.zeros((self.ntowers,self.nlevels,nlines-header))
                        w = np.zeros((self.ntowers,self.nlevels,nlines-header))
                    elif var == 'T':
                        T = np.zeros((self.ntowers,self.nlevels,nlines-header))
                    init = True
                line = line.replace('(',' ').replace(')',' ').split()
                time[tt] = float(line[0])
                if var == 'U':
                    for tw in np.arange(0,self.ntowers):
                        u[tw,:,tt] = line[(tw*3*self.nlevels)+1:((tw+1)*self.nlevels*3)+1:3]
                        v[tw,:,tt] = line[(tw*3*self.nlevels)+2:((tw+1)*self.nlevels*3)+1:3]
                        w[tw,:,tt] = line[(tw*3*self.nlevels)+3:((tw+1)*self.nlevels*3)+1:3]
                elif var == 'T':
                    for tw in np.arange(0,self.ntowers):
                        T[tw,:,tt] = line[(tw*self.nlevels)+1:((tw+1)*self.nlevels)+1]
                tt += 1
        f.close()
        self.time = time
        if var == 'U':
            self.u = u
            self.v = v
            self.w = w
        elif var == 'T':
            self.T = T

def readgrid(fdir):
    grid = open('%spointDisplacement' % fdir)
    grid.skiplines(22)
    ng = np.int(grid.realind())
    print ng


class averageprofile():
    def __init__(self,fd,varn):
        self.fdir = fd
        self.getz()
        for var in varn:
            self.getdata(var)

    def getz(self):
        f = open('%shLevelsCell' % (self.fdir),'r')
        line = f.readline().split()
        f.close()
        nz = np.shape(line)[0]
        self.nz = nz
        z = np.zeros((nz))
        for kk in range(0,nz):
            z[kk] = float(line[kk])
        self.z = z

    def getdata(self,var):
        self.nt = sum(1 for line in open('%s%s' % (self.fdir,var),'r'))
        f = open('%s%s' % (self.fdir,var),'r')
        times = np.zeros((self.nt))
        dt    = np.zeros((self.nt))
        newvar = np.zeros((self.nz,self.nt))
        for tt in range(0,self.nt):
            line = f.readline()
            if np.shape(line) != []:
                line = line.split()
                times[tt] = line[0]
                dt[tt]    = line[1]
                newvar[:,tt] = line[2:]
        f.close()
        self.time = times
        self.dt   = dt
        if var == 'U_mean':
            self.umean = newvar
        elif var == 'V_mean':
            self.vmean = newvar
        elif var == 'W_mean':
            self.wmean = newvar
        elif var == 'T_mean':
            self.Tmean = newvar
        elif var == 'Tw_mean':
            self.HFXmean = newvar
        elif var == 'q1_mean':
            self.q1mean = newvar
        elif var == 'q2_mean':
            self.q2mean = newvar
        elif var == 'q3_mean':
            self.q3mean = newvar
        elif var == 'R11_mean':
            self.r11mean = newvar
        elif var == 'R12_mean':
            self.r12mean = newvar
        elif var == 'R13_mean':
            self.r13mean = newvar
        elif var == 'R22_mean':
            self.r22mean = newvar
        elif var == 'R23_mean':
            self.r23mean = newvar
        elif var == 'R33_mean':
            self.r33mean = newvar

class errorlog():
    def __init__(self,fd):
        self.fdir = fd
        self.getErrors()

    def getErrors(self):
        f = open(self.fdir,'r')
        lines = f.readlines()
        f.close()
        time = []
        errmin = []; errmax = []; errmean = []
        cflmean = []; cflmax = []
        errflxlow = []; errflxupr = []
        errflxsth = []; errflxnth = []
        errflxest = []; errflxwst = []; errflxttl = []
        ustarmean = [];
        startreadContErr = False; errcnt = 0
        startreadFluxErr = False; errflx = 0
        grabustar = False
        for line in lines:
            if 'Time = ' in line and 'Time Step =' in line:
                time.append(np.float(line.split()[2]))
                grabustar = True
            if 'uStarMean' in line and 'iterations' in line:
                if grabustar == True:
                    line = line.split()
                    ustarmean.append(np.float(line[2]))
                    grabustar = False
            if 'Courant Number' in line:
                line = line.split()
                cflmean.append(np.float(line[3]))
                cflmax.append(np.float(line[5]))
            if 'Continuity Error' in line:
                startreadContErr = True
            if startreadContErr == True:
                if 'minimum' in line:
                    line = line.split()
                    errmin.append(np.float(line[1]))
                elif 'maximum' in line:
                    line = line.split()
                    errmax.append(np.float(line[1]))
                elif 'mean' in line:
                    line = line.split()
                    errmean.append(np.float(line[2]))
                errcnt += 1
                if errcnt == 4:
                    startreadContErr = False; errcnt = 0
            if 'Boundary Flux' in line:
                startreadFluxErr = True
            if startreadFluxErr == True:
                if 'lower' in line:
                    line = line.split()
                    errflxlow.append(np.float(line[3]))
                elif 'upper' in line:
                    line = line.split()
                    errflxupr.append(np.float(line[3]))
                elif 'south' in line:
                    line = line.split()
                    errflxsth.append(np.float(line[3]))
                elif 'north' in line:
                    line = line.split()
                    errflxnth.append(np.float(line[3]))
                elif 'east' in line:
                    line = line.split()
                    errflxest.append(np.float(line[3]))
                elif 'west' in line:
                    line = line.split()
                    errflxwst.append(np.float(line[3]))
                elif 'total' in line:
                    line = line.split()
                    errflxttl.append(np.float(line[3]))
                errflx += 1
                if errflx == 8:
                    startreadFluxErr = False; errflx = 0
        self.errtime = time; self.nt = np.shape(time)[0]
        self.cflmean = cflmean; self.cflmax = cflmax
        self.errmin = errmin; self.errmax = errmax; self.errmean = errmean
        self.errflxupr = errflxupr; self.errflxlow = errflxlow; self.errflxttl = errflxttl
        self.errflxsth = errflxsth; self.errflxnth = errflxnth; self.errflxest = errflxest; self.errflxwst = errflxwst
        self.ustarmean = ustarmean

def writePointsFile(fdir,x,y,z):
    nump = np.shape(x)[0]
    f = open('%s/points' % fdir,'w')
    f.write(r"/*--------------------------------*- C++ -*----------------------------------*\ "+"\n")
    f.write(r"| =========                 |                                                 | "+"\n")
    f.write(r"| \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox           |"+"\n")
    f.write(r"|  \\    /   O peration     | Version:  plus                                  |"+"\n")
    f.write(r"|   \\  /    A nd           | Web:      www.OpenFOAM.com                      |"+"\n")
    f.write(r"|    \\/     M anipulation  |                                                 |"+"\n")
    f.write(r"\*---------------------------------------------------------------------------*/ "+"\n")
    f.write(r"FoamFile"+"\n")
    f.write(r"{"+"\n")
    f.write(r"    version     2.0;"+"\n")
    f.write(r"    format      ascii;"+"\n")
    f.write(r"    class       vectorField;"+"\n")
    f.write(r'    location    "%s";' % fdir+"\n")
    f.write(r"    object      points;"+"\n")
    f.write(r"}"+"\n")
    f.write(r"// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //"+"\n\n\n")
    f.write('%d\n' % nump)
    f.write(r"("+"\n")
    for pp in np.arange(0,nump):
        f.write('(%f %f %f)\n' % (x[pp],y[pp],z[pp]))
    f.write(r")"+"\n\n\n")
    f.write(r"// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //")
    f.close()

def writeBoundaryFile(fdir,timedir,varn,var):
    print 'writing file %s' % varn
    if varn == 'U':
        nump = np.shape(var)[1]
    elif varn == 'T':
        nump = np.shape(var)[0]
    elif varn == 'k':
        nump = np.shape(var)[0]
    elif varn == 'qwall':
        nump = np.shape(var)[1]
    if not os.path.isdir('%s/%s' % (fdir,timedir)):
        print '%s/%s' % (fdir,timedir)
        os.makedirs('%s/%s' % (fdir,timedir))
    f = open('%s/%s/%s' % (fdir,timedir,varn),'w')
    f.write(r"/*--------------------------------*- C++ -*----------------------------------*\ "+"\n")
    f.write(r"| =========                 |                                                 | "+"\n")
    f.write(r"| \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox           |"+"\n")
    f.write(r"|  \\    /   O peration     | Version:  1.6                                   |"+"\n")
    f.write(r"|   \\  /    A nd           | Web:      www.OpenFOAM.com                      |"+"\n")
    f.write(r"|    \\/     M anipulation  |                                                 |"+"\n")
    f.write(r"\*---------------------------------------------------------------------------*/ "+"\n")
    f.write(r"FoamFile"+"\n")
    f.write(r"{"+"\n")
    f.write(r"    version     2.0;"+"\n")
    f.write(r"    format      ascii;"+"\n")
    if varn == 'T' or varn == 'k':
        f.write(r"    class       scalarAverageField;"+"\n")
    else:
        f.write(r"    class       vectorAverageField;"+"\n")
    f.write(r"    object      values;"+"\n")
    f.write(r"}"+"\n")
    f.write(r"// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //"+"\n\n")
    f.write(r"// Average"+"\n")
    if varn == 'U':
        f.write('(0 0 0)\n\n\n    %d\n' % nump)
    elif varn == 'T':
        f.write('0\n\n\n    %d\n' % nump)
    elif varn == 'k':
        f.write('0\n\n\n    %d\n' % nump)
    elif varn == 'qwall':
        f.write('(0 0 0)\n\n\n    %d\n' % nump)
    f.write(r"("+"\n")
    if varn == 'U':
        for pp in np.arange(0,nump):
            f.write('(% 12f % 12f % 12f)\n' % (var[0,pp],var[1,pp],var[2,pp]))
    elif varn == 'T':
        for pp in np.arange(0,nump):
            f.write('% 12f\n' % (var[pp]))
    elif varn == 'k':
        for pp in np.arange(0,nump):
            f.write('% 12f\n' % (var[pp]))
    elif varn == 'qwall':
        for pp in np.arange(0,nump):
            f.write('(% 12f % 12f % 12f)\n' % (var[0,pp],var[1,pp],var[2,pp]))
    f.write(r")"+"\n;\n\n")
    f.write(r"// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //")
    f.close()


def writeIdealizedBoundaryData(bdir,time,xy,yx,zz,u,v,w,T,K,z,boundaries,boundaryType):
    for bb in boundaries:
        bnddir = '%s%s' % (bdir,bb)
        if not os.path.isdir(bnddir): 
            os.makedirs(bnddir) 
        if boundaryType == 'Flat':
            if bb == 'west':
                x1,x2 = 0,0; y1,y2 = 0,-1
            if bb == 'east':
                x1,x2 = -1,-1; y1,y2 = 0,-1
            if bb == 'north':
                x1,x2 = 0,-1; y1,y2 = -1,-1
            if bb == 'south':
                x1,x2 = 0,-1; y1,y2 = 0,0
#            plt.contourf(xy,yx,zz)
#            plt.colorbar()
#            plt.scatter(xy[x1,y1],yx[x1,y1],c='r')
#            plt.scatter(xy[x2,y2],yx[x2,y2],c='b')
#            plt.title(bb)
#            plt.show()
            if bb != 'lower' and bb != 'upper':
                newx = [xy[x1,y1],xy[x2,y2]]
                newy = [yx[x1,y1],yx[x2,y2]]
                px = np.zeros((np.shape(newx)[0]*np.shape(z)[0]))
                py = np.zeros((np.shape(newx)[0]*np.shape(z)[0]))
                pz = np.zeros((np.shape(newx)[0]*np.shape(z)[0]))
                cc = 0
                for ii in np.arange(0,np.shape(newx)[0]):
                    for kk in np.arange(0,np.shape(z)[0]):
                        px[cc] = newx[ii]
                        py[cc] = newy[ii]
                        pz[cc] = z[kk]
                        cc += 1

                writePointsFile('%s' % bnddir,px,py,pz)
                U  = np.zeros((3,np.shape(u)[0]*2))
                U[0,:] = np.append(u,u)
                U[1,:] = np.append(v,v)
                U[2,:] = np.append(w,w)
                U[2,:] = 0.0
                Tf = np.append(T,T)
                Kf = np.ones(np.shape(Tf))*K
            else:
                px = np.zeros((4))
                py = np.zeros((4))
                if bb == 'lower':
                    pz = np.zeros((4)) + np.mean(zz)
                elif bb == 'upper':
                    pz = np.zeros((4)) + z[-1]
                px[0] = xy[0,0]  ; px[1] = xy[0,-1]
                px[2] = xy[-1,-1]; px[3] = xy[-1,0]
                py[0] = yx[0,0]  ; py[1] = yx[0,-1]
                py[2] = yx[-1,-1]; py[3] = yx[-1,0]
                writePointsFile('%s' % bnddir,px,py,pz)
                U  = np.zeros((3,4))
                if bb == 'lower':
                    U[0,:] = 0.0 
                    U[1,:] = 0.0 
                    U[2,:] = 0.0
                    Tf = np.zeros((4)) + T[0]
                    Kf = np.ones(np.shape(Tf))*K
                    Q  = np.zeros((3,4))
                    writeBoundaryFile('%s' % bnddir,'%d' % (time*3600),'qwall',Q)
                elif bb == 'upper':
                    U[0,:] = u[-1] 
                    U[1,:] = v[-1] 
                    U[2,:] = w[-1]
                    Tf = np.zeros((4)) + T[-1]
                    Kf = np.ones(np.shape(Tf))*K
            timedir = str(time*3600.0)
            if timedir[-2:] == '.0': 
                timedir = str(int(time*3600))
            else:
                timedir = timedir
            writeBoundaryFile('%s' % bnddir,timedir,'U',U)
            writeBoundaryFile('%s' % bnddir,timedir,'T',Tf)
            writeBoundaryFile('%s' % bnddir,timedir,'k',Kf)

        elif boundaryType == 'Terrain':
            print boundary 
    

def writeInitialConditions(fdir,u,v,w,T,z,fstr):
    nump = np.shape(u)[0]
    f = open('%sinitialValues_%s' % (fdir,fstr),'w')
    for pp in range(0,nump):
        f.write('     (%6.3f  %6.3f  %6.3f  %6.3f)\n' % (z[pp],u[pp],v[pp],T[pp]))
    f.close()


def generateInitialProfile(z,sfclay,ug,vg,Tsfc,invs,inve,invrate,freerate):
    nz = np.shape(z)[0]
    u = np.zeros((nz)); v = np.zeros((nz))
    w = np.zeros((nz)); T = np.zeros((nz))
    for kk in range(0,nz):
        if z[kk] < sfclay:
            u[kk] = ug*((1-((z[kk]-sfclay)/sfclay)**2))
            v[kk] = vg*((1-((z[kk]-sfclay)/sfclay)**2))
            T[kk] = Tsfc
        elif z[kk] >=sfclay and z[kk] < invs:
            T[kk] =Tsfc
            u[kk] = ug
            v[kk] = vg
        elif z[kk] >=invs and z[kk] < inve:
            T[kk] = T[kk-1]+(z[kk]-z[kk-1])*invrate
            u[kk] = ug
            v[kk] = vg
        else:
            T[kk] = T[kk-1]+(z[kk]-z[kk-1])*freerate
            u[kk] = ug
            v[kk] = vg
    return u,v,w,T

def generateInitialProfileWRF(fstr):
    wrfout  = ncdf('%s' % (fstr))
    hgt    = wrfout.variables['HGT'][0,:,:]
    wlat,wlon = wrfdict.latlon(wrfout)
    z,zs = wrfdict.getheight(wrfout)

    dx = wrfout.DX ; dy = wrfout.DY 

    u = wrfdict.unstagger3d(wrfout.variables['U'][0,:,:,:],2)
    v = wrfdict.unstagger3d(wrfout.variables['V'][0,:,:,:],1)
    w = wrfdict.unstagger3d(wrfout.variables['W'][0,:,:,:],0) 
    T = wrfout.variables['T'][0,:,:,:]+300.0

    avgu  = np.mean(np.mean(u,axis=2),axis=1)
    avgv  = np.mean(np.mean(v,axis=2),axis=1)
    avgw  = np.mean(np.mean(w,axis=2),axis=1)
    avgT  = np.mean(np.mean(T,axis=2),axis=1) 
    avgzs = np.mean(np.mean(zs,axis=2),axis=1)

    avgu  = np.append(np.zeros(1),avgu)
    avgv  = np.append(np.zeros(1),avgv)
    avgw  = np.append(np.zeros(1),avgw)
    avgT  = np.append(avgT[0],avgT)
    avgzs = np.append(np.zeros(1),avgzs)
    nz = np.shape(avgzs)[0]
 
    return avgzs,avgu,avgv,avgw,avgT

def generateInitialProfileSOWFA(fstr,tind):
    avgprf = averageprofile('%s' % fstr,['U_mean','V_mean','W_mean','T_mean'])
    avgu = avgprf.umean[:,tind]
    avgv = avgprf.vmean[:,tind]
    avgw = avgprf.wmean[:,tind]
    avgT = avgprf.Tmean[:,tind]
    avgzs = avgprf.z
    zsfc = avgzs[0] - 0.5*(avgzs[1]-avgzs[0])
    avgzs = avgzs - zsfc

    avgu  = np.append(np.zeros(1),avgu)
    avgv  = np.append(np.zeros(1),avgv)
    avgw  = np.append(np.zeros(1),avgw)
    avgT  = np.append(avgT[0],avgT)
    avgzs = np.append(np.zeros(1),avgzs)
    nz = np.shape(avgzs)[0]
 
    return avgzs,avgu,avgv,avgw,avgT
