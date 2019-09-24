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
import glob
import xarray as xr

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

                
def read_profiler(fpath):
    '''
    Read in the wind profiles with read_profiler('path to file')
    '''
    def _header(f):
        station_name = f.readline().split()[0]
        f.readline()
        loc  = f.readline()
        timeline = ' '.join(f.readline().split()[:2])
        time = pd.to_datetime(timeline,format='%Y-%m-%d %H:%M:%S')
        f.readline()
        angles = f.readline()
        f.readline()
        nz = int(f.readline().split()[0])
        index_str = f.readline() + 'PRF_IND'
        fix_index_str = index_str.split()
        dup_vars  = []
        dup_count = 0
        
        result = dict((i, index_str.split().count(i)) for i in index_str.split())
        for ss,varn in enumerate(fix_index_str):
            if result[varn] > 1:
                count_max = result[varn]
                if dup_count >0:fix_index_str[ss] = fix_index_str[ss]+'.'+str(dup_count)
                dup_count += 1
                if dup_count == count_max: dup_count = 0
        fix_index_str = ' '.join(fix_index_str)
        return (station_name, loc, time, angles, nz, fix_index_str)

    
    def _get_data(fpath):
        f = open(fpath,'r')
        new_profile = True
        read_file   = True
        num_profile = 0
        data = []
        while read_file:
            if new_profile:
                stn_name, location, datetime, prf_angles, nz, index_string = _header(f)
                new_profile = False
            for ll in range(0,nz):
                new_line = f.readline() + ' {}'.format(num_profile)
                data.append(new_line.split())
            end_of_profile = f.readline()
            end_of_file = f.tell()
            if f.readline() != '':
                f.seek(end_of_file)
                new_profile = True
                num_profile += 1
            else:
                read_file = False
        f.close()
        return(data, stn_name, location, datetime, prf_angles, index_string)
    
    data, stn, loc, time, angles, index_str = _get_data(fpath)
    df = pd.DataFrame(data,columns=list(index_str.split())).astype('float').replace(999.9,np.nan).astype(
                                                                {'PRF_IND': 'int64'}).sort_values(by='HT')
    df = df.round({'HT':2})
    df['Datetime'] = time
    df             = df.set_index('Datetime')
    df.station_name   = stn
    df.location       = loc
    df.profile_angles = angles
    df = df.set_index([pd.to_datetime(df.index),'HT'])
    return df


def read_pwids_mean_data(fpath):
    f = open(fpath,'rb')
    trial     = []
    date_time = []
    elap_min  = []
    wdir      = []
    wspd      = []
    u         = []
    v         = []
    TC        = []
    RH        = []
    P         = []
    f.readline()
    nt = 0
    for line in f:
        line = line.decode().replace('"."','-999.9').split(',')
        if np.size(line) > 1:
            nlvls = 5
            for nn in range(0,nlvls):
                trial.append(int(line[0]))
                date_time.append(pd.to_datetime(line[1].replace('"','') + ' ' + line[2].replace('"','')))
                elap_min.append(float(line[3]))
            wdir.append(float(line[4])); wdir.append(float(line[11])); wdir.append(float(line[18])); 
            wdir.append(float(line[25])); wdir.append(float(line[32]))
            wspd.append(float(line[5])); wspd.append(float(line[12])); wspd.append(float(line[19])); 
            wspd.append(float(line[26])); wspd.append(float(line[33]))
            u.append(float(line[6])); u.append(float(line[13])); u.append(float(line[20])); 
            u.append(float(line[27])); u.append(float(line[34]))
            v.append(float(line[7])); v.append(float(line[14])); v.append(float(line[21])); 
            v.append(float(line[28])); v.append(float(line[35]))
            TC.append(float(line[8])); TC.append(float(line[15])); TC.append(float(line[22])); 
            TC.append(float(line[29])); TC.append(float(line[36]))
            RH.append(float(line[9])); RH.append(float(line[16])); RH.append(float(line[23])); 
            RH.append(float(line[30])); RH.append(float(line[37]))
            P.append(float(line[10])); P.append(float(line[17])); P.append(float(line[24])); 
            P.append(float(line[31])); P.append(float(line[38]))
            nt += 1
    f.close()
    hgt = [2.0, 4.0, 8.0, 16.0, 32.0]*nt
    df = pd.DataFrame({'Datetime': date_time, 
                       'HT': hgt, 
                       'Trial': trial, 
                       'Elap_min': elap_min,
                       'DIR': wdir,
                       'SPD': wspd,
                       'u': u,
                       'v': v,
                       'TC': TC,
                       'RH': RH,
                       'P': P,
                      })
    df = df.set_index(['Datetime','HT']).replace(-999.9,np.nan)
    return df


def read_pwids_data(fpath):
    df = pd.read_csv(fpath,parse_dates=[['Date', 'Time']],encoding = "ISO-8859-1").replace('.',np.nan)
    df.columns = ['Datetime','Trial','Station','HT','Elap_min','DIR','SPD','u','v','TC','RH','P']
    df = df.set_index(['Datetime','HT'])
    df = df.sort_values(by='Datetime')
    df = df.astype({'DIR':'float64','SPD':'float64',
                      'u':'float64',  'v':'float64',
                     'TC':'float64', 'RH':'float64',
                      'P':'float64'})
    return df


def read_sams_data(fpath):
    '''
    SAMS data is at 10 m and there are several stations for each day. Files can be joined using:
    sams_full = []
    for pp in range(0,nfiles):
        sams_full.append(read_sams_data(file_list[pp]))
    sams_f = pd.concat(sams_full)
    In which the two indices are Datetime and Location
    '''
    f = open(fpath,'r')
    got_header = False
    while got_header == False:
        line = f.readline()
        if 'Location' in line: 
            loc = line.split()[3]
        if 'Lat' in line and 'Lon' in line: 
            lat = float(line.split()[3].replace(';',''))
            lon = float(line.split()[6])
            got_header = True
    f.close()
    df = pd.read_csv(fpath,header=None,skiprows=np.arange(0,27),names=
                     ['Date','Time','SPD','DIR','u','v','TC','RH','P','SWDown'],parse_dates=[['Date', 'Time']])
    new_cols = df.columns.values
    new_cols[0] = 'Datetime'
    df.columns = new_cols
    df = df.set_index(['Datetime'])
    df = df.sort_values(by='Datetime')
    df['Location']  = int(loc)
    df['Latitude']  = lat
    df['Longitude'] = lon
    df = df.set_index([pd.to_datetime(df.index),'Location'])
    df = df.replace(-9999.00,np.nan)
    return df    


def JRII_WRF_final_analysis(fpath, gettime='all'):
    '''
    JRII_WRF_final_analysis([path to file], times )
    times:    
              'all' - all times in the file, heights will be averaged
           datetime - it will only return the dataset for that time
    
    Read in the WRF final analysis data and put into an xarray dataset.
    
    Note: the average heights each time will change; this reader calculates
    the average height and assumes height stays constant.
    '''
    def _parse_header(f):
        '''
        Read the header and return the necessary info. Still not 100% sure
        what all of the lines in the header are...
        '''
        f.readline()
        header_start = f.tell()
        header = []
        header_len = 15
        for ll in range(0,header_len):
            line = f.readline()
            last_line = f.tell()
            header += ''.join(line)
        header = ''.join(header).split('\n')
        timeline = header[0].split()
        timestr  = pd.to_datetime('20{}{}{}:{}'.format(timeline[2],timeline[1],timeline[0],timeline[3]),
                                  format='%Y%m%d:%H')
        dimsline = header[2].split()
        nx,ny,nz = int(dimsline[0]), int(dimsline[1]), int(dimsline[2])
        
        ncols = np.shape(header[0].split())[0]
        num_zlines = int(nz/ncols)
        avgz = np.zeros(nz)
        for kk in range(0,num_zlines):
            zline = header[kk+5].split()
            [float(i) for i in zline]
            avgz[kk*ncols:ncols*(kk + 1)] = zline
        dlat    = float(header[10].split()[1])   # delat_lat
        dlon    = float(header[10].split()[0])  # delta_lon
        lat_swc = float(header[10].split()[-2]) # SW Corner
        lon_swc = float(header[10].split()[-1]) # SW Corner
        lat = np.arange(lat_swc,lat_swc+dlat*ny,dlat)
        lon = np.arange(lon_swc,lon_swc+dlon*nx,dlon)
        varlines = header[12:]
        vars4D = varlines[0].replace('NMDUMDUM','').split()
        vars3D = varlines[1].replace('NMDUMDUM','').split()
        for vline in varlines: 
            if '.' in vline: 
                # Header is bad... need to go back a line.
                f.seek(header_start)
                print('going back 1 line to {}'.format(header_start))
                for ll in range(0,header_len - 1): f.readline()
        return nx,ny,nz,ncols,avgz,vars4D,vars3D,lat,lon,timestr
    
    def _get_data(f,vars4D,vars3D,lat,lon,nx,ny,nz,ncols,hour):
        '''
        Get all of the 4D (first) and 3D (second) data for one time. Return
        a dataset.
        '''
        num4Dvars = np.shape(vars4D)[0]
        num3Dvars = np.shape(vars3D)[0]
        ds_full   = xr.Dataset()
        for vv,var in enumerate(vars4D):
            data = np.zeros((nx,ny,nz,1))
            dlinef = []
            for kk in np.arange(0,nx*ny*nz/ncols):
                dline = f.readline().split()
                dline = [float(i) for i in dline]
                dlinef.append(dline)
            dlinef = np.asarray(dlinef).flatten()
            cc = 0
            for kk in range(0,nz):
                for ii in range(0,nx):
                    for jj in range(0,ny):
                        data[ii,jj,kk] = dlinef[cc]
                        cc += 1
            da4D = xr.DataArray(data,
                                dims=('lon','lat','HT','Datetime'),
                                coords={'lon':lon, 'lat':lat, 'HT':avgz, 'Datetime':[hour]},
                                name=var)#,
                                #attrs={'units':'m/s'})
            ds_full[var] = da4D
        for vv,var in enumerate(vars3D):
            data = np.zeros((nx,ny,1))
            dlinef = []
            for kk in np.arange(0,nx*ny/ncols):
                dline = f.readline().split()
                dline = [float(i) for i in dline]
                dlinef.append(dline)
            dlinef = np.asarray(dlinef).flatten()
            cc = 0
            for ii in range(0,nx):
                for jj in range(0,ny):
                    data[ii,jj] = dlinef[cc]
                    cc += 1
            da3D = xr.DataArray(data,
                                dims=('lon','lat','Datetime'),
                                coords={'lon':lon, 'lat':lat, 'Datetime':[hour]},
                                name=var)#,
                                #attrs={'units':'m/s'})
            ds_full[var] = da3D
        return ds_full
    
    def _fill_missing_data(ds,varlist,vardims,vars3D,vars4D):
        hour = pd.Timestamp(ds.Datetime.data[0])
        missing_vars = varlist.copy()
        missing_vardims = vardims.copy()
        for vv,varn in enumerate(varlist):
            if varn in vars3D or varn in vars4D: 
                missing_vars.remove(varn)
                missing_vardims.remove(vardims[vv])
        #    if missing_vardims[vv] == 3
        nx = ds.lon.size
        ny = ds.lat.size
        nz = ds.HT.size
        for vv,varn in enumerate(missing_vars):
            if missing_vardims[vv] == 3:
                data = np.zeros((nx,ny,1))*np.nan
                da3D = xr.DataArray(data,
                                    dims=('lon','lat','Datetime'),
                                    coords={'lon':ds.lon, 'lat':ds.lat, 'Datetime':[hour]},
                                    name=varn)
                ds[varn] = da3D
            if missing_vardims[vv] == 4:
                data = np.zeros((nx,ny,nz,1))*np.nan
                da4D = xr.DataArray(data,
                                    dims=('lon','lat','HT','Datetime'),
                                    coords={'lon':ds.lon, 'lat':ds.lat, 'HT':ds.avgz, 'Datetime':[hour]},
                                    name=varn)
                ds[varn] = da4D
        return(ds)
    
    f    = open(fpath,'r')
    EOF  = False
    if gettime == 'all': ds_full = xr.Dataset()
    count = 0
    init_ds = True
    while EOF == False:
        line = f.readline()
        if 'FFFFFFFF' in line:
            nx,ny,nz,ncols,avgz,vars4D,vars3D,lat,lon,timestr = _parse_header(f)
            if gettime == 'all' or timestr==pd.to_datetime(gettime): 
                print('Getting hour: {}'.format(timestr))
            else:
                print('Searching for {}'.format(gettime))
            ds = _get_data(f,vars4D,vars3D,lat,lon,nx,ny,nz,ncols,timestr)
            if gettime == 'all': 
                if init_ds:
                    ds_full = ds.copy()
                    height_ds = ds.HT.data
                    varlist = []
                    vardims = []
                    for var in ds_full.data_vars:
                        varlist.append(var)
                        vardims.append(np.shape((ds_full[var].shape))[0])    
                    init_ds = False
                else:
                    new_height = ds.HT.data
                    height_ds = (height_ds*count + new_height)/(count+1)

                    ds_full = ds_full.assign_coords(HT=height_ds)
                    ds = ds.assign_coords(HT=height_ds)
                    ds_filled = _fill_missing_data(ds,varlist,vardims,vars3D,vars4D)
                    ds_full = xr.concat([ds_full,ds_filled],dim='Datetime')

                count+=1
            else:
                if timestr==pd.to_datetime(gettime):
                    return ds
        else:    
            f.close()
            EOF = True
    if gettime == 'all': return ds_full
    
    
def read_radiosonde(fpath):
    '''
    Read in the radiosonde data and convert to pandas df.
    The sonde only has 1 time, so the time index will need
    to be created and will be the same for each row.
    '''
    def _header(header_list):
        elev = float(header_list[0][0]); lat = float(header_list[0][1]); lon = float(header_list[0][2])
        launch_time = pd.to_datetime('{} {}, 20{} {}:{}'.format(
                      header_list[1][4],header_list[1][5],header_list[1][3],
                      header_list[1][7],header_list[1][8]))
        header = header_list[2]
        header = ['HT','P','TC','RH','Td','MR','DIR','SPD','Type']
        return(lat,lon,elev,launch_time,header)
            
    f = open(fpath,'r')
    sonde_f = f.readlines()
    f.close()
    for kk in sonde_f:
        if np.shape(kk.split())[0] == 0:
            sonde_f.remove(kk)
    for ll,line in enumerate(sonde_f):
        if 'MANDATORY LEVELS' in line:
            man_lev_ind = ll
        if 'SIGNIFICANT LEVELS' in line:
            sig_lev_ind = ll
        sonde_f[ll] = line.split()
    header_lines = 3
    header_list = sonde_f[:header_lines]
    sonde_dat = np.array(sonde_f[header_lines:man_lev_ind])
    sonde_man = sonde_f[man_lev_ind+1:sig_lev_ind]
    sonde_sig = sonde_f[sig_lev_ind+1:]
    lat,lon,elev,launch_time,header = _header(header_list)
    sonde_df = pd.DataFrame(sonde_dat,columns=header)
    sonde_df['Datetime'] = pd.Timestamp(launch_time)
    sonde_df = sonde_df.astype(dtype={'HT':'float64','P':'float64','TC':'float64','RH':'float64','Td':'float64',
                                      'MR':'float64','DIR':'float64','SPD':'float64'})
    sonde_df['HT'] = sonde_df['HT'] - elev
    sonde_df['Lat'] = lat; sonde_df['Lon'] = lon
    sonde_df = sonde_df.set_index(['Datetime','HT'])
    sonde_df['Theta'] = (sonde_df.TC+273.15)*((1000.0/sonde_df.P)**0.286)
    return sonde_df
    
 