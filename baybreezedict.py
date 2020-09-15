'''
  What it does: This dictionary contains functions for detecting
                and validating bay breezes.

  Who made it: patrick.hawbecker@nrel.gov
  When: 01/07/20
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
from collections import Counter
#from sklearn.linear_model import LinearRegression
import scipy.stats as stats
from mmctools.helper_functions import calc_wind

def spatial_breeze_check(onshore_min,
                         onshore_max,
                         wrfout,
                         land_mask=None,
                         dT_calc='vertical',
                         dT_cutoff_pct=75,
                         wdir_check='vertical',
                         wdir_cutoff_pct=80):
    
    if land_mask is None:
        land_mask = wrfout.LANDMASK
    vel10,dir10 = calc_wind(wrfout,u='U10',v='V10')
    vel10 = vel10.where(land_mask == 1.0)
    dir10 = dir10.where(land_mask == 1.0)
    nx = len(wrfout.west_east)
    ny = len(wrfout.south_north) 
    onshore_winds = dir10.where((dir10 >= onshore_min) & (dir10 <= onshore_max))
    #onshore_winds /= onshore_winds
    onshore_winds = onshore_winds.fillna(0.0)
    
    top_ind = 18
    bot_ind = 0
    if (wdir_check != 'vertical') & (wdir_check != 'smoothed'):
        print('wdir_check must be vertical or smoothed... defaulting to vertical')
        wdir_check = 'vertical'
    dU = vel10.where(~np.isnan(onshore_min))
    if wdir_check == 'smoothed':
        smooth_dir = dir10.copy()
        h_window = 25
        
    temp = np.squeeze(wrfout.T)
    if dT_calc == 'vertical':
        bb_temp = temp.copy().where(~np.isnan(onshore_min))
        z_f = (np.squeeze(wrfout.PH) + np.squeeze(wrfout.PHB))/9.8 - np.squeeze(wrfout.HGT)
        zs_f = 0.5*(z_f[1:,:,:]+z_f[:-1,:,:])   
        dT = (bb_temp[top_ind,:,:] - bb_temp[bot_ind,:,:]) / (zs_f[top_ind,:,:] - zs_f[bot_ind,:,:])
    elif dT_calc == 'horizontal':
        t2 = wrfout.T2.where(land_mask == 1.0)
        dT = t2.where(~np.isnan(onshore_min))
        
        
        
    if (wdir_check == 'smoothed') or (dT_calc == 'horizontal'):
        for ii in np.arange(window_start_i,nx-window_start_i):
            for jj in np.arange(window_start_j,ny-window_start_j):
                if wdir_check == 'smoothed':
                    if ((ii >= h_window) & (jj >= h_window)) & ((ii <= nx-h_window) & (jj <= ny-h_window)):
                        dir_window = dir10.data[jj-h_window:jj+h_window+1,ii-h_window:ii+h_window+1].copy()
                        if ~np.all(np.isnan(dir_window)):
                            dir_window_range = np.nanmax(dir_window) - np.nanmin(dir_window)
                            if dir_window_range > 300.0:
                                if np.nanmedian(dir_window) < 180.0:
                                    with np.errstate(invalid='ignore'):
                                        dir_window[dir_window >= 270.0] -= 360.0
                                else:
                                    with np.errstate(invalid='ignore'):
                                        dir_window[dir_window <= 90.0] += 360.0

                            smooth_dir[jj,ii] = np.nanmean(dir_window)
                        else:
                            smooth_dir[jj,ii] = np.nan
                    else:
                        smooth_dir[jj,ii] = np.nan
                if dT_calc == 'horizontal':
                    if ~np.isnan(dT[jj,ii]):
                        T_window = t2[jj-1:jj+2,ii-1:ii+2].data.flatten()
                        dT_window = t2[jj,ii].data - T_window
                        dT_window[np.where(dT_window>=0.0)] = np.nan
                        if np.count_nonzero(np.isnan(dT_window)) > 8:
                            dT[jj,ii] = np.nan
                        else:
                            dT[jj,ii] = np.nanmean(dT_window)



        
    if wdir_check == 'smoothed':        
        wdir_cutoff = np.nanpercentile(smooth_dir - dir10,wdir_cutoff_pct)
    else: # wdir_check == 'vertical'
        u = wrfout.U[top_ind,:,:].data
        v = wrfout.V[top_ind,:,:].data

        u = 0.5*(u[:,1:] + u[:,:-1])
        v = 0.5*(v[1:,:] + v[:-1,:])
        wdir1km = 180. + np.degrees(np.arctan2(u, v))
        wdir_cutoff = np.nanpercentile(wdir1km - dir10,wdir_cutoff_pct)
    dT_cutoff = np.max([0.0,np.nanpercentile(dT,dT_cutoff_pct)])


    good_wind_dir = onshore_winds.copy()
    diff_wind_dir = onshore_winds.copy()
    window_start_i = min(np.where(~np.isnan(onshore_min))[1])
    window_start_j = min(np.where(~np.isnan(onshore_min))[0])
    for ii in np.arange(window_start_i,nx-window_start_i):
        for jj in np.arange(window_start_j,ny-window_start_j):
            
            if ~np.isnan(dU[jj,ii]):
                U_window = vel10[jj-1:jj+2,ii-1:ii+2].data.flatten()
                dU_window = vel10[jj,ii].data - U_window
                with np.errstate(invalid='ignore'):
                    dU_window[np.where(dU_window<=0.0)] = np.nan
                if np.count_nonzero(np.isnan(dU_window)) > 8:
                    dU[jj,ii] = np.nan
                else:
                    dU[jj,ii] = np.nanmean(dU_window)

            if onshore_winds[jj,ii] > 0.0:
                if wdir_check == 'smoothed':
                    meso_onshore = (smooth_dir[jj,ii] > onshore_min[jj,ii]) & (smooth_dir[jj,ii] < onshore_max[jj,ii])
                    meso_wind  = smooth_dir[jj,ii]
                else:
                    meso_onshore = (wdir1km[jj,ii] > onshore_min[jj,ii]) & (wdir1km[jj,ii] < onshore_max[jj,ii])
                    meso_wind = wdir1km[jj,ii]
                local_wind = dir10[jj,ii]
                if (meso_wind > 0.0) and (local_wind < 0.0):
                    print(meso_wind,local_wind)
                    local_wind += 360.0
                if (local_wind > 0.0) and (meso_wind < 0.0):
                    print(meso_wind,local_wind)
                    meso_wind += 360.0
                #wind_diff = np.abs(meso_wind - local_wind)
                wind_diff = meso_wind - local_wind
                diff_wind_dir[jj,ii] = wind_diff
                is_different = (wind_diff >= wdir_cutoff)

                if ~is_different and meso_onshore:
                    good_wind_dir[jj,ii] = 0.0
                    
    
    
    bay_breeze_area = good_wind_dir.copy()
    bay_breeze_area_data = bay_breeze_area.where(land_mask==1.0).data
    bay_breeze_area_data = bay_breeze_area_data*0.0
    bay_breeze_area_data[good_wind_dir > 0.0] += 1.0
    bay_breeze_area_data[dT >= dT_cutoff] += 1.0
    bay_breeze_area_data[dU > 0.5] += 1.0
    bay_breeze_area.data = bay_breeze_area_data 
    
    bay_breeze_detection_dict = {   'breeze':bay_breeze_area,
                                 'good_wdir':good_wind_dir,
                                        'dT':dT,
                                        'dU':dU}
    
    
    
    for kk,key in enumerate(bay_breeze_detection_dict.keys()):
        if key == 'breeze':
            var = bay_breeze_detection_dict['good_wdir'].copy()
            var.data = bay_breeze_detection_dict[key]
        else:
            var = bay_breeze_detection_dict[key]
            var.name = key
        if kk == 0:
            ds = xr.Dataset({key: var})
        else:
            ds = xr.merge([ds,var])

    ds['dT_cutoff'] = dT_cutoff
    ds['wdir_cutoff'] = wdir_cutoff            
    ds = ds.expand_dims('datetime')
    dtime = ds.XTIME.expand_dims('datetime')
    ds = ds.drop('XTIME')
    ds['datetime'] = dtime

    return(ds)



class DetectBayBreeze():
    def __init__(self, station, 
                 inland      = None,
                 resample    = True,
                 sample_rate = '60min',
                 light_winds = 3.08667,
                 show_plot   = False, 
                 method      = 'StaufferThompson2015',
                 min_points  = 3,
                 verbose     = False):

        if (inland is None) & (method != 'Stauffer2015'):
            raise ValueError('Must specify an inland station ("inland=") with this method.')

        self.detected  = False
        self.validated = False
        self.analyzed  = False
        
        num_station_pts = np.min([station.wspd.dropna(dim='datetime',how='any').size,
                                 station.wdir.dropna(dim='datetime',how='any').size])
        num_inland_pts  = np.min([inland.wspd.dropna(dim='datetime',how='any').size,
                                  inland.wdir.dropna(dim='datetime',how='any').size])
        if (num_station_pts >= min_points):
            self.analyzed = True
            if verbose: print('We have enough points. detecting a wind shift...')
            self._detect_wind_shift(station,resample,sample_rate,light_winds,show_plot,verbose)
            
            if self.wind_shift:
                case_date = pd.to_datetime(np.squeeze(self.passage.data))

                if method == 'Stauffer2015':
                    
                    if self.wind_shift:
                        self.detected  = True
                        self._detect_dwpt_change(station,resample,sample_rate,show_plot,verbose)
                        if self.dwpt_change:
                            if verbose: print('bay breeze day')
                            self.validated = True
                            print('bay breeze validated for {}/{}/{}'.format(
                                   case_date.month,case_date.day,case_date.year))                        
                elif method == 'Sikora2010':
                    self._detect_temp_change(station,resample,sample_rate,show_plot,verbose)
                    self._detect_gust(station,resample,sample_rate,show_plot,verbose)
                    self._detect_precip(station,resample=resample,sample_rate=sample_rate,
                                        show_plot=show_plot,verbose=verbose,method=method)
                    self._detect_clouds(station,resample=resample,sample_rate=sample_rate,
                                        show_plot=show_plot,verbose=verbose,method=method)
                    if (self.wind_shift) & (self.temp_change) & (self.burst_or_increase) & \
                       (not self.measured_precip) & (not self.clouds_detected):
                        if verbose: print('bay breeze day')
                        self.detected  = True
                        if (num_inland_pts >= min_points):
                            self._inland_compare(station,inland,resample=resample,sample_rate=sample_rate,
                                                show_plot=show_plot,verbose=verbose,method=method)
                            print('onshore inland: {}'.format(self.onshore_inland))
                            if (not self.onshore_inland):
                                print('bay breeze validated for {}/{}/{}'.format(
                                       case_date.month,case_date.day,case_date.year))
                                self.validated = True

                     
                elif method == 'StaufferThompson2015':
                    self._detect_precip(station,resample=resample,sample_rate=sample_rate,
                                        show_plot=show_plot,verbose=verbose,method=method)
                    self._detect_clouds(station,resample=resample,sample_rate=sample_rate,
                                        show_plot=show_plot,verbose=verbose,method=method)
                                                
                    if (self.wind_shift) & (not self.measured_precip) & (not self.clouds_detected):
                        if verbose: print('bay breeze day')
                        self.detected  = True
                        if (num_inland_pts >= min_points):
                            self._inland_compare(station,inland,resample=resample,sample_rate=sample_rate,
                                                show_plot=show_plot,verbose=verbose,method=method)
                            if not self.onshore_inland:
                                print('bay breeze validated for {}/{}/{}'.format(
                                       case_date.month,case_date.day,case_date.year))
                                self.validated = True                        
                        
        if (show_plot) & (self.detected):
            if method != 'StaufferThompson2015':
                fig,ax = plt.subplots(nrows=3,sharex=True,figsize=(8,8))
            else:
                fig,ax = plt.subplots(nrows=2,sharex=True,figsize=(8,5))
                
            wspd_plot = station.wspd.dropna(how='all',dim='datetime').resample(
                                        datetime=sample_rate).interpolate('linear')
            wdir_plot = station.wdir.dropna(how='all',dim='datetime').resample(
                                        datetime=sample_rate).interpolate('linear')
            wspd_inpl = inland.wspd.dropna(how='all',dim='datetime')
            wdir_inpl = inland.wdir.dropna(how='all',dim='datetime')
            wspd_plot.plot(ax=ax[0],marker='o',c='blue',label=str(station.station.values))
            wdir_plot.plot(ax=ax[1],marker='o',c='darkblue')
            
            try:
                n_wspd_inpl = len(np.asarray(np.squeeze(wspd_inpl.data)))
            except TypeError:
                n_wspd_inpl = 1 
            try:
                n_wdir_inpl = len(np.asarray(np.squeeze(wdir_inpl.data)))
            except TypeError:
                n_wdir_inpl = 1
                
            if n_wspd_inpl > 1:
                plt_data = wspd_inpl.resample(datetime=sample_rate).interpolate('linear')
                if len(plt_data) > 1:
                    plt_data.plot(ax=ax[0],marker='o',c='green',label=inland.station.values)
            if n_wdir_inpl > 1:
                plt_data = wdir_inpl.resample(datetime=sample_rate).interpolate('linear')
                if len(plt_data) > 1:
                    plt_data.plot(ax=ax[1],marker='o',c='darkgreen')
                
            ax[0].fill_between([station.datetime.data[0],
                  station.datetime.data[-1]],0.0,light_winds,
                  color='grey',alpha=0.2,lw=0.0)
            ax[1].fill_between([station.datetime.data[0],
                  station.datetime.data[-1]],float(station.onshore_min.values),
                  float(station.onshore_max.values),color='grey',alpha=0.2,lw=0.0)
            ax[0].legend(frameon=False)
            ax[0].axvline(pd.to_datetime(self.passage.values),c='k',ls=':',alpha=0.5)
            ax[1].axvline(pd.to_datetime(self.passage.values),c='k',ls=':',alpha=0.5)
            ax[1].set_xlim(station.datetime.data[0],station.datetime.data[-1])
            ax[0].set_ylim(0,15)
            ax[1].set_ylim(0,360)
            if self.validated:
                fill_color = 'darkgreen'
            else:
                fill_color = 'darkred'
            ax[0].fill_betweenx(np.arange(0,100),pd.to_datetime(self.start.values),
                    pd.to_datetime(self.end.values),alpha=0.1, color=fill_color)
            ax[1].fill_betweenx(np.arange(0,360),pd.to_datetime(self.start.values),
                    pd.to_datetime(self.end.values),alpha=0.1, color=fill_color)
            
            if method != 'StaufferThompson2015':
                if method == 'Stauffer2015':  
                    third_plot = station.dwpt.sel(datetime=slice(self.start,self.end)).dropna(
                                                    dim='datetime',how='all')
                if method == 'Sikora2010': 
                    third_plot = station.temp.sel(datetime=slice(self.start,self.end)).dropna(
                                                    dim='datetime',how='all')
                try:
                    plt_len = len(np.squeeze(third_plot.data))
                except:
                    plt_len = len(third_plot.data)
                if plt_len > 1:
                    third_plot = third_plot.resample(datetime=sample_rate).interpolate('linear')

#                    third_plot = station.dwpt.dropna(how='all',dim='datetime')#.resample(
#                                                            #datetime=sample_rate).interpolate('linear')
#                    third_plot = station.temp.dropna(how='all',dim='datetime')#.resample(
#                                                            #datetime=sample_rate).interpolate('linear')
                #print(third_plot)
                #print(len(third_plot))
                if len(third_plot.data) > 1:
                    third_plot.plot(ax=ax[2],marker='o',c='blue')
                    ax[2].set_ylim(np.min(third_plot)-1,np.max(third_plot)+1)
                #print(pd.to_datetime(self.start.values),pd.to_datetime(self.end.values))
                try:
                    ax[2].fill_betweenx(np.arange(-50,50),pd.to_datetime(self.start.values),
                                        pd.to_datetime(self.end.values),alpha=0.1, color=fill_color) 
                except:
                    print('Unable to fill...')
                try:
                    ax[2].axvline(pd.to_datetime(self.passage.values),c='k',ls=':',alpha=0.5)
                except:
                    print('Unable to draw line...')
            plt.show()


        



    def _detect_wind_shift(self, station,
                           resample    = True, # Currently forcing to resample...
                           sample_rate = '60min',
                           light_winds = 3.08667,
                           show_plot   = False, 
                           verbose     = False):

        wspd = station.wspd.dropna(dim='datetime',how='all').resample(datetime=sample_rate).interpolate('linear')
        wdir = station.wdir.dropna(dim='datetime',how='all').resample(datetime=sample_rate).interpolate('linear')

        is_offshore = (wdir>station.onshore_max) | (wdir<station.onshore_min) & (wdir <= 360.0)
        is_light    = ((wspd<light_winds) & (wdir > 360.0)) | (wspd < 1.0)
        # Condition 1: Offshore winds, light and variable or calm (less than light_winds)
        offshore_conditions = is_light | is_offshore
        is_onshore  = ~is_offshore
        is_greater_than_0 = wspd > light_winds
        onshore_conditions = is_onshore & is_greater_than_0
        if verbose:
            print('Offshore conditions met: {}'.format(np.any(offshore_conditions).data))
            print('Onshore conditions met: {}'.format(np.any(onshore_conditions).data))

        wind_shift          = False
        bay_breeze_detected = False
        bay_breeze_start    = None
        bay_breeze_pass     = None
        bay_breeze_end      = None
        if np.any(offshore_conditions) and np.any(onshore_conditions):
            offshore = ['']*wdir.size
            onshore  = ['']*wdir.size
            lbl = 'a'
            offshore_flag = False
            for ii in range(1,wdir.size):
                if offshore_conditions[ii-1]:
                    if offshore_conditions[ii]:
                        offshore[ii-1] = lbl
                        offshore[ii] = lbl
                    elif onshore_conditions[ii]:
                        onshore[ii] = lbl
                elif onshore_conditions[ii-1]:
                    if onshore_conditions[ii]:
                        onshore[ii] = lbl
                    elif offshore_conditions[ii]:
                        lbl = chr(ord(lbl)+1)
                        offshore[ii] = lbl
            if verbose:
                for ll in zip(offshore,onshore):
                    print(ll)

                    
            offshore_count = Counter(offshore) 
            onshore_count  = Counter(onshore)

            for lbl in offshore_count.keys():
                offshore_time = 0.0
                onshore_time  = 0.0
                if len(lbl) > 0:
                    if lbl in onshore_count.keys():
                        
                        offshore_inds = np.where(np.asarray(offshore)==lbl)[0]
                        offshore_s    = offshore_inds[0]
                        offshore_e    = offshore_inds[-1]

                        onshore_inds  = np.where(np.asarray(onshore)==lbl)[0]
                        onshore_s     = onshore_inds[0]
                        onshore_e     = onshore_inds[-1]
                        offshore_e = onshore_s

                        if len(offshore_inds) == 1:
                            offshore_time = 0.0
                        else:
                            offshore_start = wspd.datetime.isel(datetime=offshore_s).data
                            offshore_end   = wspd.datetime.isel(datetime=offshore_e).data
                            offshore_time = np.timedelta64(offshore_end-offshore_start,'m') / np.timedelta64(1, 'h')

                        if len(onshore_inds) == 1:
                            onshore_time = 0.0
                        else:
                            onshore_start = wspd.datetime.isel(datetime=onshore_s).data
                            onshore_end   = wspd.datetime.isel(datetime=onshore_e).data
                            onshore_time = np.timedelta64(onshore_end-onshore_start,'m') / np.timedelta64(1, 'h')
                    if verbose: print('Label {} offshore: {} hours'.format(lbl,offshore_time))
                    if verbose: print('Label {} onshore:  {} hours'.format(lbl,onshore_time))
                    if offshore_time >= 1.0 and onshore_time >= 2.0:
                        bay_breeze_start = wdir.datetime[np.where(np.asarray(offshore)==lbl)[0][0]]
                        bay_breeze_pass  = wdir.datetime[np.where(np.asarray(onshore)==lbl)[0][0]]
                        bay_breeze_end   = wdir.datetime[np.where(np.asarray(onshore)==lbl)[0][-1]]
                        if bay_breeze_start < bay_breeze_pass:
                            wind_shift = True

        if verbose:
            if wind_shift:
                print('wind shift detected.')
            else:
                print('No wind shift detected.')
        self.wind_shift = wind_shift
        self.start    = bay_breeze_start
        self.passage  = bay_breeze_pass
        self.end      = bay_breeze_end 
                                
            
    def _detect_dwpt_change(self, station,
                           resample    = True, # Currently forcing to resample...
                           sample_rate = '60min',
                           show_plot   = False, 
                           verbose     = False):
        dwpt = station.dwpt.sel(datetime=slice(self.start,self.end)).dropna(
                                dim='datetime',how='all')
        try:
            dwpt_len = len(np.squeeze(dwpt.data))
        except:
            dwpt_len = len(dwpt.data)

        if dwpt_len > 1:
            dwpt = dwpt.resample(datetime=sample_rate).interpolate('linear')
            try:
                dwpt_before = np.nanmin(dwpt.sel(datetime=slice(self.passage - pd.Timedelta(0.5,'h'),
                                                                self.passage)).data)
            except:
                dwpt_before = 999.9
            try:
                dwpt_after  = np.nanmax(dwpt.sel(datetime=slice(self.passage,
                                                                self.passage + pd.Timedelta(1.0,'h'))).data)
            except:
                dwpt_after = -999.9

            if dwpt_before - dwpt_after <= -1.0:
                self.dwpt_change = True
            else:
                self.dwpt_change = False
            if verbose: print('Increase in dewpoint over 1˚C: {}'.format(self.dwpt_change))
        else:
            self.dwpt_change = False

            
            
    def _detect_temp_change(self, station,
                           resample    = True, # Currently forcing to resample...
                           sample_rate = '60min',
                           show_plot   = False, 
                           verbose     = False):
    
        temp = station.temp.sel(datetime=slice(self.start,self.end)).dropna(
                                dim='datetime',how='all')
        try:
            temp_len = len(np.squeeze(temp.data))
        except:
            temp_len = len(temp.data)

        if temp_len > 1:
            temp = temp.resample(datetime=sample_rate).interpolate('linear')
            try:
                temp_before = temp.sel(datetime=slice(self.passage - pd.Timedelta(0.5,'h'),
                                                      self.passage - pd.Timedelta(1,'m'))).data[-1]
            except:
                temp_before = -999.9
            try:
                temp_after  = temp.sel(datetime=slice(self.passage,self.passage + pd.Timedelta(1.0,'h'))).data[1]
            except:
                temp_after = 999.9

            if temp_before - temp_after >= 0.0:
                self.temp_change = True
            else:
                self.temp_change = False
            if verbose: print('Increase in dewpoint over 1˚C: {}'.format(self.temp_change))
        else:
            self.temp_change = False
    
    
        '''
        temp = station.temp.sel(datetime=slice(self.start,self.end)).dropna(
                                dim='datetime',how='all').resample(
                                datetime=sample_rate).interpolate('linear')
        temp_before = temp.sel(datetime=slice(self.passage - pd.Timedelta(0.5,'h'),self.passage)).data[-2]
        temp_after  = temp.sel(datetime=slice(self.passage,self.passage + pd.Timedelta(1.0,'h'))).data[1]
        
        if temp_before - temp_after >= 0.0:
            self.temp_change = True
        else:
            self.temp_change = False
        if verbose: print('Decrease or leveling of temperature: {}'.format(self.temp_change))
        '''

    def _detect_gust(self, station,
                     resample    = True, # Currently forcing to resample...
                     sample_rate = '60min',
                     show_plot   = False, 
                     verbose     = False):
        wspd = station.wspd.dropna(dim='datetime',how='all').resample(datetime=sample_rate).interpolate('linear')
        wspd_before = wspd.sel(datetime=slice(self.passage - pd.Timedelta(0.5,'h'),self.passage)).data[-2]
        wspd_after  = wspd.sel(datetime=slice(self.passage,self.passage + pd.Timedelta(1.0,'h')))
        wspd_gust   = np.max(wspd.sel(datetime=slice(self.passage,self.passage + pd.Timedelta(0.5,'h'))).data)
        if wspd_gust > wspd_before:
            burst = True
        else:
            burst = False
        
        x = np.arange(0,len(wspd_after))
        slope = stats.linregress(x, wspd_after.data)[0]

        #slope = LinearRegression().fit(x.reshape(-1, 1), wspd_after.data.reshape(-1, 1)).coef_[0][0]
        if slope > 0.0:
            wspd_increase = True
        else:
            wspd_increase = False
        
        if burst or wspd_increase:
            self.burst_or_increase = True
        else:
            self.burst_or_increase = False  
        if verbose: print('Short burst or constant increase in wind speed: {}'.format(self.burst_or_increase))

            
    def _detect_precip(self, station,
                       resample    = True, # Currently forcing to resample...
                       sample_rate = '60min',
                       method      = 'StaufferThompson2015',
                       show_plot   = False, 
                       verbose     = False):
        var_names = []
        for dd in station.data_vars: var_names.append(dd)
        if 'pcip' in var_names:
            if (type(station.pcip.data[0]) == str):
                pcp = station.pcip.dropna(dim='datetime',how='any')
                station_pcp = ['']*len(pcp)
                for vv,val in enumerate(np.squeeze(pcp.data)):
                    if val == 'NP': 
                        station_pcp[vv] = 0.0
                    elif ('R' in val) or ('S' in val):
                        print(val)
                        station_pcp[vv] = 1.0
                    else:
                        station_pcp[vv] = 0.0
                station_pcp = np.asarray(station_pcp)
                station_pcp = xr.DataArray(station_pcp,dims=['datetime'],coords=[station.datetime.values])
            else:
                station_pcp = station.pcip#.dropna(dim='datetime',how='any')
        elif ('rainc' in var_names) and ('rainnc' in var_names):
            rainc  = np.squeeze(station.rainc.data)
            rainnc = np.squeeze(station.rainnc.data)
            tot_rain = (rainc[1:] - rainc[:-1]) + (rainnc[1:] - rainnc[:-1])
            tot_rain = np.concatenate([np.asarray([0.0]),tot_rain])

            station_pcp = xr.DataArray(tot_rain,dims=['datetime'],coords=[station.datetime.values])
            station_pcp = station_pcp.dropna(dim='datetime',how='any')        
#        station_pcp = station_pcp.dropna(dim='datetime',how='all').resample(
#                                         datetime=sample_rate).interpolate('linear')
        station_pcp = station_pcp.fillna(value=0.0).resample(datetime=sample_rate).interpolate('linear')
        if method == 'Sikora2010':
            station_pcp = station_pcp.sel(datetime=slice(self.passage - pd.Timedelta(6,'h'),
                                                         self.passage))
        if len(station_pcp.values) > 0:
            total_precip = np.sum(station_pcp.values)
        else:
            total_precip = 0.0   
        if total_precip > 0.0:
            self.measured_precip = True
        else:
            self.measured_precip = False
        if verbose: print('No measured precipitation: {}'.format(not self.measured_precip))
    
    
    def _detect_clouds(self, station,
                       resample    = True, # Currently forcing to resample...
                       sample_rate = '60min',
                       method      = 'StaufferThompson2015',
                       show_plot   = False, 
                       verbose     = False):
        
        var_names = []
        for dd in station.data_vars: var_names.append(dd)
        if 'cldc' in var_names:
            station_cld = station.cldc.dropna(dim='datetime',how='any')
        elif ('skyc1' in var_names) and ('skyc2' in var_names) and ('skyc3' in var_names) and ('skyc4' in var_names):
            skyc1 = station.skyc1.values#.dropna(dim='datetime',how='any')
            station_cld = ['']*len(skyc1)

            for vv,val in enumerate(skyc1):
                station_cld[vv] = '{},{},{},{}'.format(str(val).strip(),
                                                       str(station.skyc2.values[vv]).strip(),
                                                       str(station.skyc3.values[vv]).strip(),
                                                       str(station.skyc4.values[vv]).strip())
                if 'nan' in station_cld[vv]: station_cld[vv] = 'nan'
                if station_cld[vv] == ',,,': station_cld[vv] = 'nan'
                if station_cld[vv] == 'VV,,,': station_cld[vv] = 'nan'
            station_cld = xr.DataArray(station_cld,dims=['datetime'],coords=[station.datetime.values])
        elif ('clw' in var_names):
            clw = station.clw.dropna(dim='datetime',how='any')
            station_cld = ['']*len(clw)
            for vv,val in enumerate(np.squeeze(clw.data)):
                if val <= 0.03: 
                    station_cld[vv] = 'CLR'
                else:
                    station_cld[vv] = 'OVC'
            station_cld = np.asarray(station_cld)

            if verbose: print('CLR: {}; OVC: {}'.format(
                len(np.where(station_cld == 'CLR')[0]),
                len(np.where(station_cld == 'OVC')[0])))
            station_cld = xr.DataArray(station_cld,dims=['datetime'],coords=[station.datetime.values])
        else:
            station_cld = ['CLR']*len(station.datetime.dropna(dim='datetime',how='any'))
            print(len(station_cld),len(station.datetime.values))
            station_cld = xr.DataArray(station_cld,dims=['datetime'],coords=[station.datetime.values])
                
        for cc,cloud in enumerate(station_cld.values):
            if ',' in cloud: 
                if 'OVC' in cloud:
                    station_cld.values[cc] = 'OVC'
                elif 'BKN' in cloud:
                    station_cld.values[cc] = 'BKN'
                elif 'SCT' in cloud:
                    station_cld.values[cc] = 'SCT'
                elif 'FEW' in cloud:
                    station_cld.values[cc] = 'FEW'
                elif 'CLR' in cloud:
                    station_cld.values[cc] = 'CLR'
        ctype_count = Counter(station_cld.values)
        cloud_sum = 0
        nobs = 0
        # Cloud cover is reported in terms of 1/8th of sky cover with 1-2/8th being FEW,...
        # ...3-4/8ths being SCT, 5-7/8th being BKN and 8/8 denoted at OVC
        cloud_val = {'CLR': 0.0, 'FEW': 1.5, 'SCT': 3.5, 'BKN': 6.0, 'OVC': 8.0}
        for ctype in ctype_count.keys():
            if ctype != '' and ctype != 'nan' and ctype != ' ' and ctype !='A7:':
                cloud_sum += ctype_count[ctype]*cloud_val[ctype]
                nobs += ctype_count[ctype]
            if verbose: print(ctype,ctype_count[ctype])
        if nobs == 0: 
            cloudy_skies = True
        else:
            avg_cloud = cloud_sum/nobs
            for cloud, val in cloud_val.items():
                if val == avg_cloud:
                    avg_cloud_type = cloud

            #cloudy_skies = avg_cloud >= cloud_val['BKN']
            cloudy_skies = avg_cloud >= 5.0 # Define categories - BKN is from 5.0 to 7.0, OVC is anything above
        self.clouds_detected = cloudy_skies
        if verbose: print('No clouds detected: {}'.format(not self.clouds_detected))

    def _inland_compare(self, station, inland,
                        resample    = True, # Currently forcing to resample...
                        sample_rate = '60min',
                        method      = 'StaufferThompson2015',
                        light_winds = 3.08667,
                        show_plot   = False, 
                        verbose     = False):    
        '''
        Sikora just looks for onshore winds at inland site.
        Stauffer and Thompson check for winds not onshore OR wind speeds less than 3.0 m/s.
        '''

        wspd = inland.wspd.dropna(dim='datetime',how='all').resample(datetime=sample_rate).interpolate('linear')
        wdir = inland.wdir.dropna(dim='datetime',how='all').resample(datetime=sample_rate).interpolate('linear')
        
        wspd = wspd.sel(datetime=slice(self.passage,self.end))
        wdir = wdir.sel(datetime=slice(self.passage,self.passage+pd.Timedelta(1.5,'h')))

        is_offshore = (wdir>station.onshore_max) | (wdir<station.onshore_min) & (wdir <= 360.0)
        is_onshore  = ~is_offshore
        any_onshore = np.any(is_onshore.data)
        if method == 'StaufferThompson2015':
            inland_winds_with_onshore = np.squeeze(wspd.data)[np.where(np.squeeze(is_onshore.data))]
            weak_winds = np.all(inland_winds_with_onshore <= light_winds)
            if (not any_onshore) or weak_winds: 
                self.onshore_inland = False
            else:
                self.onshore_inland = True
        else:
            self.onshore_inland = any_onshore

        onshore_count = 0
        if any_onshore:
            for vv in np.squeeze(is_onshore.data):
                if vv: onshore_count += 1
            
        if verbose:
            print('Inland wind direction is onshore: {}'.format(any_onshore))
            if np.any(is_onshore.data):
                print('Number of onshore measurements: {0} out of {1} ({2:2.2f}%)'.format(
                       onshore_count,len(np.squeeze(is_onshore.data)),
                       100.0*(onshore_count/len(np.squeeze(is_onshore.data)))))
            


            

class BayBreezeDetection():
    '''
    Detect bay breeze events with several different methods. Validate the events
    with inland station data.

    Methods implemented: "StaufferThompson2015"
                         "Stauffer2015"
                         "Sikora2010" (not yet available...)
    '''
    def __init__(self,station,inland_stations,resample=False,sample_rate='60min',light_winds=3.08667,show_plot=False,
                 method='StaufferThompson2015',min_points=3,verbose=False):
        '''
        Variables:
               station: dataset for given station
        inland_station: dataset for inland station used in verification
              resample: Should the data be resampled (boolean)
           sample_rate: if 'resample' is True, resample to this rate (str)
           light_winds: 6 knots in m/s
                method: StaufferThompson2015, Stauffer2015, or Sikora2010 (not available)
             show_plot: plot the inland stations (boolean)
            min_points: minimum number of data points required for analysis (int)
               verbose: Prints statements going through each step (boolean)
        '''
        bay_breeze_detected  = False
        bay_breeze_validated = False

        if (station.wspd.dropna(dim='datetime',how='any').size >= min_points) & \
           (station.wdir.dropna(dim='datetime',how='any').size >= min_points):
            if verbose: print('Detecting wind shift.')
            self._detect_wind_shift(station,resample=resample,sample_rate=sample_rate,verbose=verbose)
            if self.wind_shift:
                var_names = []
                for dd in station.data_vars: var_names.append(dd)
                if 'cldc' in var_names:
                    station_cld = station.cldc.dropna(dim='datetime',how='any')
                elif ('skyc1' in var_names) and ('skyc2' in var_names) and ('skyc3' in var_names) and ('skyc4' in var_names):
                    skyc1 = station.skyc1.values#.dropna(dim='datetime',how='any')
                    station_cld = ['']*len(skyc1)

                    for vv,val in enumerate(skyc1):
                        station_cld[vv] = '{},{},{},{}'.format(str(val).strip(),
                                                               str(station.skyc2.values[vv]).strip(),
                                                               str(station.skyc3.values[vv]).strip(),
                                                               str(station.skyc4.values[vv]).strip())
                        if 'nan' in station_cld[vv]: station_cld[vv] = 'nan'
                        if station_cld[vv] == ',,,': station_cld[vv] = 'nan'
                        if station_cld[vv] == 'VV,,,': station_cld[vv] = 'nan'
                    station_cld = xr.DataArray(station_cld,dims=['datetime'],coords=[station.datetime.values])
                elif ('clw' in var_names):
                    clw = station.clw.dropna(dim='datetime',how='any')
                    station_cld = ['']*len(clw)
                    for vv,val in enumerate(np.squeeze(clw.data)):
                        if val <= 0.03: 
                            station_cld[vv] = 'CLR'
                        else:
                            station_cld[vv] = 'OVC'
                    station_cld = np.asarray(station_cld)
                    
                    print('CLR: {}; OVC: {}'.format(
                        len(np.where(station_cld == 'CLR')[0]),
                        len(np.where(station_cld == 'OVC')[0])))
                    station_cld = xr.DataArray(station_cld,dims=['datetime'],coords=[station.datetime.values])
                else:
                    station_cld = ['CLR']*len(station.datetime.dropna(dim='datetime',how='any'))
                    print(len(station_cld),len(station.datetime.values))
                    station_cld = xr.DataArray(station_cld,dims=['datetime'],coords=[station.datetime.values])
                
                if 'pcip' in var_names:
                    if (type(station.pcip.data[0]) == str):
                        pcp = station.pcip.dropna(dim='datetime',how='any')
                        station_pcp = ['']*len(pcp)
                        for vv,val in enumerate(np.squeeze(pcp.data)):
                            if val == 'NP': 
                                station_pcp[vv] = 0.0
                            elif ('R' in val) or ('S' in val):
                                print(val)
                                station_pcp[vv] = 1.0
                            else:
                                station_pcp[vv] = 0.0
                        station_pcp = np.asarray(station_pcp)
                        station_pcp = xr.DataArray(station_pcp,dims=['datetime'],coords=[station.datetime.values])
                    else:
                        station_pcp = station.pcip.dropna(dim='datetime',how='any')
                elif ('rainc' in var_names) and ('rainnc' in var_names):
                    rainc  = np.squeeze(station.rainc.data)
                    rainnc = np.squeeze(station.rainnc.data)
                    tot_rain = (rainc[1:] - rainc[:-1]) + (rainnc[1:] - rainnc[:-1])
                    tot_rain = np.concatenate([np.asarray([0.0]),tot_rain])
                    
                    station_pcp = xr.DataArray(tot_rain,dims=['datetime'],coords=[station.datetime.values])
                    station_pcp = station_pcp.dropna(dim='datetime',how='any')
                    
                if len(station_cld.dropna(dim='datetime',how='any')) > 0:
                    print('evaluating clouds')
                    self._check_clouds_and_precip(station_cld,station_pcp)
                if self.clear_or_dry:
                    if method=='Stauffer2015':
                        self._check_1degDwptRise(station,resample=resample,sample_rate=sample_rate,
                                                 show_plot=show_plot)
                        if self.dwpt_rise:
                            print('BAY BREEZE DETECTED')
                            bay_breeze_detected = True
                    else:
                        print('BAY BREEZE DETECTED')
                        bay_breeze_detected = True

        self.detected  = bay_breeze_detected
        self.validated = bay_breeze_validated

        if self.detected:
            self._validate_bay_breeze(station,inland_stations=inland_stations,resample=resample,
                                      sample_rate=sample_rate,show_plot=show_plot)

    def _check_1degDwptRise(self, station, resample=True, sample_rate=False, 
                            show_plot=False):
        _, index = np.unique(station['datetime'], return_index=True)
        station = station.isel(datetime=index)
        if resample:
            station_dpt = station.dwpt.dropna(dim='datetime',how='any').resample(
                                            datetime=sample_rate).interpolate('linear')
        else:
            station_dpt = station.dwpt.dropna(dim='datetime',how='any')
        if len(station_dpt.values) > 1:
            window_size = pd.to_timedelta(1,unit='h')
            dpt_before = station_dpt.sel(datetime=slice(str(self.passage.values-window_size),str(self.passage.values)))
            dpt_after  = station_dpt.sel(datetime=slice(str(self.passage.values),str(self.passage.values+window_size)))
            print('Checking dewpoint rise...')
            if (len(dpt_before.values) > 0) and (len(dpt_after.values) > 0):
                print('here...')
                min_before = np.around(np.min(dpt_before.values),decimals=4)
                max_after  = np.around(np.max(dpt_after.values),decimals=4)
                if max_after - min_before >= 1.0:
                    print('{} - {}: good!'.format(max_after,min_before))
                    self.dwpt_rise = True
                else:
                    print('{} - {}: bad.'.format(max_after,min_before))
                    self.dwpt_rise = False
            else:
                print('Not enough datapoints...')
                self.dwpt_rise = False
        else:
            print('Not enough datapoints...')
            self.dwpt_rise = False
        
        if show_plot:
            station_dpt.plot.line(marker='o')
            plt.axvline(pd.to_datetime(self.passage.values),c='k',ls=':',alpha=0.5)
            plt.title(station['datetime'].values[0])
            plt.show()
        


    def _validate_bay_breeze(self, station, inland_stations, light_winds=3.08667, 
                             resample=True, sample_rate=False, show_plot=False):
        station_names = inland_stations['station'].values
        n_inland = np.squeeze(np.shape(station_names))
        
        if np.size(n_inland) == 0:
            n_inland = 1
            station_names = [station_names]
        station_validated = np.zeros((n_inland), dtype=bool)
        bay_breeze_validated  = False
        low_winds_in_period   = False
        onshore_dir_validated = False
        if show_plot: fig,ax = plt.subplots(nrows=2,sharex=True)
        for sin,inland in enumerate(station_names):
            if n_inland > 1:
                inland_wspd = inland_stations.sel(station=inland).wspd.dropna(dim='datetime',how='any')
                inland_wdir = inland_stations.sel(station=inland).wdir.dropna(dim='datetime',how='any') 
            else:
                inland_wspd = inland_stations.wspd.dropna(dim='datetime',how='any')
                inland_wdir = inland_stations.wdir.dropna(dim='datetime',how='any')
            if resample:
                inland_wspd = inland_stations.sel(station=inland).wspd.dropna(dim='datetime',how='any').resample(
                                                                datetime=sample_rate).interpolate('linear')
                inland_wdir = inland_stations.sel(station=inland).wdir.dropna(dim='datetime',how='any').resample(
                                                                datetime=sample_rate).interpolate('linear')
            
            is_onshore  = ((inland_wdir>=station.onshore_min) & (inland_wdir <= station.onshore_max)).data.squeeze()
            onshore  = ['']*inland_wdir.size
            on_lbl = 'a'
            new_on_lbl = True

            if inland_wdir.size > 1:
                for ii in range(0,inland_wdir.size):
                    if is_onshore[ii]:
                        onshore[ii] = '{}'.format(on_lbl)
                        new_on_lbl = False
                    else:
                        if new_on_lbl == False:
                            on_lbl = chr(ord(on_lbl)+1)
                            new_on_lbl = True
            onshore_count = Counter(onshore)
            onshore_time  = 0.0
            #print(onshore)
            for lbl in onshore_count.keys():
                if len(lbl) > 0:
                    if onshore_count[lbl] > 1:
                        onshore_inds  = np.where(np.asarray(onshore) == lbl)[0]
                        onshore_s     = inland_wdir.datetime.data[onshore_inds[0]]
                        onshore_e     = inland_wdir.datetime.data[onshore_inds[-1]]
                        onshore_start = inland_wspd.datetime.sel(datetime=onshore_s, method='nearest').data
                        onshore_end   = inland_wspd.datetime.sel(datetime=onshore_e, method='nearest').data
                        #onshore_start = inland_wspd.datetime.sel(datetime=onshore_s, method='nearest', tolerance=np.timedelta64(1,'h')).data
                        #onshore_end   = inland_wspd.datetime.sel(datetime=onshore_e, method='nearest', tolerance=np.timedelta64(1,'h')).data
                        #print(np.timedelta64(onshore_end-onshore_start,'m'))
                        onshore_time += np.timedelta64(onshore_end-onshore_start,'m') / np.timedelta64(1, 'h')
                        #print(onshore_time)
            if onshore_time <= 2.0:
                #print('Good time...')
                onshore_dir_validated = True
            else:
                onshore_dir_validated = False
            low_winds_in_period = np.all(inland_wspd.data <= light_winds)
            #print(onshore_dir_validated,low_winds_in_period)

            # Wind speed validation... less than light_winds the whole period:

            if onshore_dir_validated | low_winds_in_period:
                station_validated[sin] = True
            else:
                print('Onshore dir: {}, onshore low winds: {}'.format(onshore_dir_validated,low_winds_in_period))
            if inland_wspd.sizes['datetime'] > 0 and show_plot:
                inland_wspd.plot(marker='o',ax=ax[0],label=inland)
                inland_wdir.plot(marker='o',ax=ax[1],label=inland)
                ax[0].fill_between([pd.to_datetime('2000'),
                                    pd.to_datetime('2030')],0.0,light_winds,color='grey',alpha=0.2,lw=0.0)
                ax[1].fill_between([pd.to_datetime('2000'),
                                    pd.to_datetime('2030')],float(station.onshore_min.values),float(station.onshore_max.values),
                                    color='grey',alpha=0.2,lw=0.0)
                ax[1].set_xlim(station.datetime.data[0],station.datetime.data[-1])

        if show_plot:
            ax[0].legend(loc=2)
            ax[0].set_ylim(0,15)
            ax[1].set_ylim(0,360)
            plt.show()

        if np.all(station_validated):
            print("Bay Breeze Validated!")
            bay_breeze_validated = True
        else:
            print('Cannot validate Bay Breeze...')
            bay_breeze_validated = False

        self.validated = bay_breeze_validated

    def _check_clouds_and_precip(self,station_cld,station_pcp):
        '''
        If skies were less than broken and there was no measureable rainfall during the day
        '''
        for cc,cloud in enumerate(station_cld.values):
            if ',' in cloud: 
                if 'OVC' in cloud:
                    station_cld.values[cc] = 'OVC'
                elif 'BKN' in cloud:
                    station_cld.values[cc] = 'BKN'
                elif 'SCT' in cloud:
                    station_cld.values[cc] = 'SCT'
                elif 'FEW' in cloud:
                    station_cld.values[cc] = 'FEW'
                elif 'CLR' in cloud:
                    station_cld.values[cc] = 'CLR'
        ctype_count = Counter(station_cld.values)
        cloud_sum = 0
        nobs = 0
        # Cloud cover is reported in terms of 1/8th of sky cover with 1-2/8th being FEW,...
        # ...3-4/8ths being SCT, 5-7/8th being BKN and 8/8 denoted at OVC
        cloud_val = {'CLR': 0.0, 'FEW': 1.5, 'SCT': 3.5, 'BKN': 6.0, 'OVC': 8.0}
        for ctype in ctype_count.keys():
            if ctype != '' and ctype != 'nan' and ctype != ' ':
                cloud_sum += ctype_count[ctype]*cloud_val[ctype]
                nobs += ctype_count[ctype]
        if nobs == 0: 
            cloudy_skies = True
        else:
            avg_cloud = cloud_sum/nobs
            for cloud, val in cloud_val.items():
                if val == avg_cloud:
                    avg_cloud_type = cloud

            #cloudy_skies = any('BKN' in cloud for cloud in clouds.values) | any('OVC' in cloud for cloud in clouds.values)
            cloudy_skies = avg_cloud >= cloud_val['BKN']
        clear_skies  = not cloudy_skies
        print('clear skies: ', clear_skies)
        #print(station_pcp.datetime)
        #print(self.passage)

        if len(station_pcp.values) > 0:
            total_precip = np.sum(station_pcp.values)
        else:
            total_precip = 0.0   
        if total_precip > 0.0:
            rain = True
        else:
            rain = False
        dry = not rain    
        print('dry: ', dry)
        clear_or_dry = clear_skies | dry
        self.clear_or_dry = clear_or_dry


    def _detect_wind_shift(self, station, sample_rate='60min', light_winds=3.08667, resample=True,verbose=False):
        '''
        From Stauffer and Thompson, 2015:

        For each day, the daytime (0900 to 1600 Eastern Standard Time, EST) wind directions were evaluated: 
            
        If the hourly wind direction measurement changed from either offshore, calm, or light and variable (less than 6 kt), 
        to onshore sustained for two or more consecutive hours during the period...
        '''

        _, index = np.unique(station['datetime'], return_index=True)
        station = station.isel(datetime=index)
        if resample:
            station_spd = station.wspd.dropna(dim='datetime',how='all').resample(
                                            datetime=sample_rate).interpolate('linear').dropna(dim='datetime',how='all')
            station_dir = station.wdir.dropna(dim='datetime',how='all').resample(
                                            datetime=sample_rate).interpolate('linear').dropna(dim='datetime',how='all')
        else:
            station_spd = station.wspd.dropna(dim='datetime',how='any')
            station_dir = station.wdir.dropna(dim='datetime',how='any')
        
        
        
        
        
        fig,ax = plt.subplots(nrows=2,figsize=(12,8),sharex=True)
        station_spd.plot.line(marker='o',ax=ax[0])
        ax[0].set_ylim(0,15)
        ax[0].fill_between([pd.to_datetime('2000'),
                          pd.to_datetime('2030')],0.0,light_winds,color='grey',alpha=0.2,lw=0.0)
        station_dir.plot.line(color='purple', marker='o',ax=ax[1])
        ax[1].fill_between([station.datetime.data[0],
                          station.datetime.data[-1]],float(station.onshore_min.values),
                          float(station.onshore_max.values),color='grey',alpha=0.2,lw=0.0)
        #ax[1].set_xlim(station.datetime.data[0],station.datetime.data[-1])
        #ax[1].set_ylim(0,360)
        ax[1].set_xlim(station.datetime.data[0],station.datetime.data[-1])
        plt.show()
        
        
        
        
        
        
        
        is_onshore  = ((station_dir>=station.onshore_min) & (station_dir <= station.onshore_max)
                        & (station_dir <= 360.0)).data.squeeze()
        is_offshore = ((station_dir<station.onshore_min) | \
                      ((station_dir>station.onshore_max) & (station_dir<=360.0))).data.squeeze()
        
        is_lt_vrb   = ((station_spd<light_winds) & (station_dir > 360.0)).data.squeeze()
        if len(is_lt_vrb) == len(is_offshore):
            offshore_conditions = is_lt_vrb | is_offshore
        else:
            offshore_conditions = False
        if verbose: print('   offshore: {}'.format(is_offshore))
        if verbose: print('light & vrb: {}'.format(is_lt_vrb))
        if verbose: print(' eaither or: {}'.format(offshore_conditions))
        if verbose: print('    onshore: {}'.format(is_onshore))
        wind_shift          = False
        bay_breeze_detected = False
        bay_breeze_start    = None
        bay_breeze_pass     = None
        bay_breeze_end      = None
        if np.any(offshore_conditions) and np.any(is_onshore):
            offshore = ['']*station_dir.size
            onshore  = ['']*station_dir.size
            off_lbl = 'a'
            on_lbl = 'a'

            new_off_lbl = False
            new_on_lbl = True
            for ii in range(0,station_dir.size):
                if is_onshore[ii]:
                    onshore[ii] = '{}'.format(on_lbl)
                    new_on_lbl = False
                    if new_off_lbl == False:
                        off_lbl = chr(ord(off_lbl)+1)
                        new_off_lbl = True

                elif offshore_conditions[ii]:
                    offshore[ii] = '{}'.format(off_lbl)
                    new_off_lbl = False
                    if new_on_lbl == False:
                        on_lbl = chr(ord(on_lbl)+1)
                        new_on_lbl = True
                else:
                    if offshore_conditions[ii-1]:  
                        off_lbl = chr(ord(off_lbl)+1)
                        on_lbl = off_lbl
                        new_off_lbl = False
                    elif is_onshore[ii-1]:
                        on_lbl = chr(ord(on_lbl)+1)
                        new_on_lbl = True

            if verbose: 
                print(offshore)
                print(onshore)
            offshore_count = Counter(offshore) 
            onshore_count  = Counter(onshore)
            for lbl in offshore_count.keys():
                if len(lbl) > 0:
                    if lbl in onshore_count.keys():
                        if resample:
                            offshore_time = (offshore_count[lbl])*pd.to_timedelta(sample_rate) / np.timedelta64(1, 'h')
                            onshore_time  = (onshore_count[lbl]-1)*pd.to_timedelta(sample_rate) / np.timedelta64(1, 'h')
                        else:
                            offshore_inds = np.where(np.asarray(offshore)==lbl)[0]
                            offshore_s    = offshore_inds[0]
                            offshore_e    = offshore_inds[-1]

                            onshore_inds  = np.where(np.asarray(onshore)==lbl)[0]
                            onshore_s     = onshore_inds[0]
                            onshore_e     = onshore_inds[-1]
                            offshore_e = onshore_s

                            if len(offshore_inds) == 1:
                                offshore_time = 0.0
                            else:
                                offshore_start = station_spd.datetime.isel(datetime=offshore_s).data
                                offshore_end   = station_spd.datetime.isel(datetime=offshore_e).data
                                offshore_time = np.timedelta64(offshore_end-offshore_start,'m') / np.timedelta64(1, 'h')

                            if len(onshore_inds) == 1:
                                onshore_time = 0.0
                            else:
                                onshore_start = station_spd.datetime.isel(datetime=onshore_s).data
                                onshore_end   = station_spd.datetime.isel(datetime=onshore_e).data
                                onshore_time = np.timedelta64(onshore_end-onshore_start,'m') / np.timedelta64(1, 'h')
                        if verbose: print('Label {} offshore: {} hours'.format(lbl,offshore_time))
                        if verbose: print('Label {} onshore:  {} hours'.format(lbl,onshore_time))
                        if offshore_time >= 1.0 and onshore_time >= 1.8:
                            bay_breeze_start = station_spd.datetime[np.where(np.asarray(offshore)==lbl)[0][0]]
                            bay_breeze_pass  = station_spd.datetime[np.where(np.asarray(onshore)==lbl)[0][0]]
                            bay_breeze_end   = station_spd.datetime[np.where(np.asarray(onshore)==lbl)[0][-1]]
                            #print(bay_breeze_start.data)
                            #print(bay_breeze_pass.data)
                            #print(bay_breeze_end.data)
                            if bay_breeze_start < bay_breeze_pass:
                                wind_shift = True
        if (wind_shift) and (verbose): print('wind shift detected...')
        self.wind_shift = wind_shift
        self.start    = bay_breeze_start
        self.passage  = bay_breeze_pass 
        self.end      = bay_breeze_end  
