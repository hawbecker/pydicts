'''
  What it does: This dictionary contains functions for detecting
                and validating bay breezes.

  Who made it: hawbecke@ucar.edu
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
import scipy.stats as stats
from mmctools.helper_functions import calc_wind
import skimage.morphology
from matplotlib.colors import Normalize
from matplotlib.patches import Ellipse

def spatial_breeze_check(onshore_min,
                         onshore_max,
                         out_file,
                         land_mask=None,
                         dT_calc='vertical',
                         dT_cutoff_pct=75,
                         wdir_check='vertical',
                         wdir_cutoff_pct=80,
                         model='WRF',
                         top_ind = 18,
                         check_rain=False,
                         rain_da=None,
                         rain_cutoff=0.25, # trace amount of rain
                         check_clouds=False,
                         cloud_cutoff=0.5):
    
    if model == 'WRF':
        if land_mask is None: land_mask = out_file.LANDMASK
        t2   = out_file.T2.where(land_mask == 1.0)
        u10  = out_file.U10.where(land_mask == 1.0)
        v10  = out_file.V10.where(land_mask == 1.0)
        sfcP = out_file.PSFC.where(land_mask == 1.0)
        dx   = out_file.DX/1000.0
        dy   = out_file.DY/1000.0
    elif model == 'HRRR':
        if land_mask is None: land_mask  = out_file.LAND_P0_L1_GLC0
        t2   = out_file.TMP_P0_L103_GLC0.where(land_mask == 1.0)
        u10  = out_file.UGRD_P0_L103_GLC0.sel(lv_HTGL2=10.0)
        v10  = out_file.VGRD_P0_L103_GLC0.sel(lv_HTGL2=10.0)
        sfcP = out_file.PRES_P0_L1_GLC0.where(land_mask == 1.0)
        dx,dy = 3.0,3.0
        

    vel10 = (u10**2 + v10**2)**0.5
    dir10 = 180. + np.degrees(np.arctan2(u10, v10))

    vel10 = vel10.where(land_mask == 1.0)
    dir10 = dir10.where(land_mask == 1.0)
    nx,ny = np.shape(land_mask)
    x = np.arange(0,nx)*dx
    y = np.arange(0,nx)*dy
    xy,yx = np.meshgrid(x,y)

    onshore_winds = dir10.where((dir10 >= onshore_min) & (dir10 <= onshore_max))
    #onshore_winds /= onshore_winds
    onshore_winds = onshore_winds.fillna(0.0)
    
    bot_ind = 0
    if (wdir_check != 'vertical') & (wdir_check != 'smoothed'):
        print('wdir_check must be vertical or smoothed... defaulting to vertical')
        wdir_check = 'vertical'
    dU = vel10.where(~np.isnan(onshore_min))
    if wdir_check == 'smoothed':
        smooth_dir = dir10.copy()
        
    if dT_calc == 'vertical':
        if model == 'WRF': 
            temp = np.squeeze(out_file.T)
            bb_temp = temp.copy().where(land_mask==1.0)
            z_f = (np.squeeze(out_file.PH) + np.squeeze(out_file.PHB))/9.8 - np.squeeze(out_file.HGT)
            zs_f = 0.5*(z_f[1:,:,:]+z_f[:-1,:,:])
            zs_f_avg = np.nanmean(zs_f,axis=(1,2))
            dist_to_1km = np.abs(zs_f_avg - 1000.0)
            top_ind = np.where(dist_to_1km == np.nanmin(dist_to_1km))[0][0]
            z_top = zs_f[top_ind,:,:]
            z_bot = zs_f[bot_ind,:,:]
            bb_temp_top = bb_temp[top_ind,:,:]
            bb_temp_bot = bb_temp[bot_ind,:,:]
        elif model == 'HRRR':
            z_top = out_file.HGT_P0_L100_GLC0.sel(lv_ISBL5=85000.0)
            z_bot = 2.0
            bb_temp_top = out_file.TMP_P0_L100_GLC0.where(land_mask==1.0).sel(lv_ISBL0=92500.0)*((1000.0/925.0)**0.286)
            bb_temp_bot = out_file.POT_P0_L103_GLC0.where(land_mask==1.0) # 2m Pot. Temp.
            

        dT = (bb_temp_top - bb_temp_bot) / (z_top - z_bot)
    elif dT_calc == 'horizontal':
        dT = t2.where(~np.isnan(onshore_min))
        
    window_start_i = min(np.where(~np.isnan(onshore_min))[1])
    window_start_j = min(np.where(~np.isnan(onshore_min))[0])
    h_window = np.round(window_start_i/2)
        
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

    dT_full = dT.copy()
    
    if wdir_check == 'smoothed':        
        wdir_cutoff = np.nanpercentile(smooth_dir - dir10,wdir_cutoff_pct)
    else: # wdir_check == 'vertical'
        if model == 'WRF': 
            u = out_file.U[top_ind,:,:].data
            v = out_file.V[top_ind,:,:].data
            u_top = 0.5*(u[:,1:] + u[:,:-1])
            v_top = 0.5*(v[1:,:] + v[:-1,:])
        elif model == 'HRRR':
            u_top = out_file.UGRD_P0_L100_GLC0.sel(lv_ISBL1=85000.0)
            v_top = out_file.VGRD_P0_L100_GLC0.sel(lv_ISBL1=85000.0)

        wdir1km = 180. + np.degrees(np.arctan2(u_top, v_top))
        wdir_cutoff = np.nanpercentile(np.abs(wdir1km - dir10),wdir_cutoff_pct)
        
    wdir_cutoff = np.max([10.0,wdir_cutoff]) # Lower limit
    wdir_cutoff = np.min([100.0,wdir_cutoff])# Upper limit

    good_wind_dir = onshore_winds.copy()
    diff_wind_dir = onshore_winds.copy()
    
    test_meso  = np.zeros(np.shape(onshore_winds))*np.nan
    test_local = np.zeros(np.shape(onshore_winds))*np.nan
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
                wind_diff = np.abs(meso_wind - local_wind)
                diff_wind_dir[jj,ii] = wind_diff
                is_different = (wind_diff >= wdir_cutoff)

                if ~is_different and meso_onshore:
                    good_wind_dir[jj,ii] = 0.0
                test_meso[jj,ii]  = meso_wind
                test_local[jj,ii] = local_wind
                    
    
    
    bay_breeze_area = good_wind_dir.copy()
    bay_breeze_area_data = bay_breeze_area.where(land_mask==1.0).data
    bay_breeze_area_data = bay_breeze_area_data*0.0
    bay_breeze_area_data[good_wind_dir > 0.0] += 1.0
    
    near_shore_dT = dT.where(~np.isnan(onshore_min))
    if model == 'WRF':
        inland_dT = dT.where(np.isnan(onshore_min) & (land_mask == 1)).sel(south_north=slice(window_start_j,ny - window_start_j),
                                                                             west_east=slice(window_start_i,ny - window_start_i))
    elif model == 'HRRR':
        inland_dT = dT.where(np.isnan(onshore_min) & (land_mask == 1)).sel(ygrid_0=slice(window_start_j,ny - window_start_j),
                                                                           xgrid_0=slice(window_start_i,ny - window_start_i))
        
    print('mean near shore: {}\ninland: {}'.format(
            np.nanmean(near_shore_dT),
            np.nanmean(inland_dT)))
    print('pct near shore: {}\ninland: {}'.format(
            np.nanpercentile(near_shore_dT,dT_cutoff_pct),
            np.nanpercentile(inland_dT,dT_cutoff_pct)))
    
    if (np.nanpercentile(dT,dT_cutoff_pct) > 0.0):
        print('Whole domain is stable.. checking near shore')
        
        if (np.nanpercentile(near_shore_dT,dT_cutoff_pct) > 0.0):
            print('Near shore also stable...')
            dT_cutoff = np.nan
        else:
            print('Near shore not stable. Finding dT_cutoff for just this region.')
            dT_cutoff = np.max([0.0,np.nanpercentile(near_shore_dT,dT_cutoff_pct)])
    else:
        print('Whole domain is neutral or convective. Calculating dT_cutoff over whole domain.')
        dT_cutoff = np.max([0.0,np.nanpercentile(dT,dT_cutoff_pct)])
        
    print('cutoff all: {}'.format(dT_cutoff))


    dT = dT.where(good_wind_dir > 0)#.fillna(-999.9)
    dU = dU.where(good_wind_dir > 0).fillna(-999.9)

    dT = dT.fillna(-999.9)
    bay_breeze_area_data[dT >= dT_cutoff] += 1.0
    #bay_breeze_area_data[dU > 0.5] += 1.0
    bay_breeze_area.data = bay_breeze_area_data
    
    bay_breeze_detection_dict = {   'breeze':bay_breeze_area,
                                 'good_wdir':good_wind_dir,
                                        'dT':dT,
                                   'dT_full':dT_full,
                                        'dU':dU
                                }
                                
    
    if check_rain:
        if rain_da is None:
            if model == 'HRRR':
                try:
                    rain_da = out_file.APCP_P8_L1_GLC0_acc
                except:
                    rain_da = out_file.APCP_P8_L1_GLC0_acc1h
            else:
                var_list = list(out_file.variables)
                rain_vars = []
                for varn in var_list: 
                    if 'RAIN' in varn: 
                        rain_vars.append(varn)
                if len(rain_vars) > 0:
                    for vv,varn in enumerate(rain_vars):
                        if vv == 0:
                            rain_da = out_file[varn].copy()
                        else:
                            rain_da += out_file[varn]
                else:
                    print('No rain var detected... setting to 0')
                    rain_da = out_file.T*0.0
     
        
        is_raining = rain_da.where(rain_da>=rain_cutoff)
        is_raining /= is_raining
        is_raining = is_raining.fillna(0.0)
        bay_breeze_detection_dict['is_raining'] = is_raining
        bay_breeze_detection_dict['rain'] = rain_da

    if check_clouds:
        if model == 'WRF':
            clouds = out_file.CLDFRA.max(axis=0)
        elif model == 'HRRR':
            clouds = out_file.TCDC_P0_L10_GLC0/100.0
        is_cloudy = clouds.where(clouds >= cloud_cutoff)
        is_cloudy /= is_cloudy
        is_cloudy = is_cloudy.fillna(0.0)
        bay_breeze_detection_dict['is_cloudy'] = is_cloudy
        bay_breeze_detection_dict['clouds'] = clouds



    
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
    if check_clouds:
        ds['cloud_cutoff'] = cloud_cutoff
    if check_rain:
        ds['rain_cutoff'] = rain_cutoff

    ds = ds.expand_dims('datetime')
    if model == 'WRF':
        dtime = ds.XTIME.expand_dims('datetime')
        ds = ds.drop('XTIME')
    elif model == 'HRRR':
        try:
            forecast_time = pd.to_timedelta(out_file.TMP_P0_L103_GLC0.forecast_time,'h')
        except:
            forecast_time = pd.to_timedelta(0,'h')
        dtime = xr.DataArray([pd.to_datetime(out_file.TMP_P0_L103_GLC0.initial_time,format='%m/%d/%Y (%H:%M)')],dims=['datetime'])
        dtime += forecast_time
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
                 verbose     = False,
                 remove_cldORprecip = False):

        if (inland is None) & (method != 'Stauffer2015'):
            raise ValueError('Must specify an inland station ("inland=") with this method.')

        self.detected     = False
        self.validated    = False
        self.analyzed     = False
        self.cldsORprecip = False
        
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
                            if verbose: print('onshore inland: {}'.format(self.onshore_inland))
                            if (not self.onshore_inland):
                                print('bay breeze validated for {}/{}/{}'.format(
                                       case_date.month,case_date.day,case_date.year))
                                self.validated = True

                     
                elif method == 'StaufferThompson2015':
                    self._detect_precip(station,resample=resample,sample_rate=sample_rate,
                                        show_plot=show_plot,verbose=verbose,method=method)
                    self._detect_clouds(station,resample=resample,sample_rate=sample_rate,
                                        show_plot=show_plot,verbose=verbose,method=method)
                    self.cldsORprecip =  not (not self.measured_precip) and (not self.clouds_detected)
                    if verbose:
                        print('clouds: ',self.clouds_detected)
                        print('precip: ',self.measured_precip)
                        print('Clouds or precip for {}/{}/{}: {}'.format(
                                           case_date.month,case_date.day,case_date.year,self.cldsORprecip))
                    if remove_cldORprecip:
                        move_to_validate = self.wind_shift & (not self.cldsORprecip)
                    else:
                        move_to_validate = self.wind_shift
                        
                    if move_to_validate:                        
                        if verbose: print('bay breeze day')
                        self.detected  = True
                        if (num_inland_pts >= min_points):
                            self._inland_compare(station,inland,resample=resample,sample_rate=sample_rate,
                                                show_plot=show_plot,verbose=verbose,method=method)
                            if not self.onshore_inland:
                                print('bay breeze validated for {}/{}/{}'.format(
                                       case_date.month,case_date.day,case_date.year))
                                self.validated = True
                            else:
                                print('Inland check failed for {}/{}/{}'.format(
                                       case_date.month,case_date.day,case_date.year))
                        
        if (show_plot) & (self.detected):
            if method != 'StaufferThompson2015':
                fig,ax = plt.subplots(nrows=3,sharex=True,figsize=(8,8))
            else:
                fig,ax = plt.subplots(nrows=2,sharex=True,figsize=(8,5))
            
            if len(station.wspd.dropna(how='all',dim='datetime')) > 1:
                wspd_plot = station.wspd.dropna(how='all',dim='datetime').resample(
                                            datetime=sample_rate).interpolate('linear')
                wspd_plot.plot(ax=ax[0],marker='o',c='blue',label=str(station.station.values))
                station.wspd.plot(ax=ax[0],marker='o',c='blue',alpha=0.3)
            if len(station.wdir.dropna(how='all',dim='datetime')) > 1:
                wdir_plot = station.wdir.dropna(how='all',dim='datetime').resample(
                                            datetime=sample_rate).interpolate('linear')
                wdir_plot.plot(ax=ax[1],marker='o',c='darkblue')
                station.wdir.plot(ax=ax[1],marker='o',c='darkblue',alpha=0.3)
            
            wspd_inpl = inland.wspd.dropna(how='all',dim='datetime')
            wdir_inpl = inland.wdir.dropna(how='all',dim='datetime')
            
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
                    wspd_inpl.plot(ax=ax[0],marker='o',c='green',alpha=0.3)
            if n_wdir_inpl > 1:
                plt_data = wdir_inpl.resample(datetime=sample_rate).interpolate('linear')
                if len(plt_data) > 1:
                    plt_data.plot(ax=ax[1],marker='o',c='darkgreen')
                    wdir_inpl.plot(ax=ax[1],marker='o',c='darkgreen',alpha=0.3)
                
            ax[0].fill_between([station.datetime.data[0],
                  station.datetime.data[-1]],0.0,light_winds,
                  color='grey',alpha=0.2,lw=0.0)
            ax[1].fill_between([station.datetime.data[0],
                  station.datetime.data[-1]],float(station.onshore_min.values),
                  float(station.onshore_max.values),color='grey',alpha=0.2,lw=0.0)
            ax[0].legend(frameon=False)
            if self.detected:
                ax[0].axvline(pd.to_datetime(self.passage.values),c='k',ls=':',alpha=0.5)
                ax[1].axvline(pd.to_datetime(self.passage.values),c='k',ls=':',alpha=0.5)
            ax[1].set_xlim(station.datetime.data[0],station.datetime.data[-1])
            ax[0].set_ylim(0,15)
            ax[1].set_ylim(0,360)
            
            if self.detected:
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
        is_light    = ((wspd<light_winds) & (wdir > 360.0)) | (wspd < light_winds)
        # Condition 1: Offshore winds, light and variable or calm (less than light_winds)
        offshore_conditions = is_light | is_offshore
        is_onshore  = ~is_offshore
        is_greater_than_0 = wspd > 0.0
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
            off_lbl = 'a'
            on_lbl  = 'a'
            lbl = 'a'
            offshore_flag = False
            onshore_flag  = False
            for ii in range(0,wdir.size):
                if offshore_conditions[ii]:
                    offshore_flag = True
                    offshore[ii] = off_lbl
                    if onshore_flag == True:
                        onshore_flag = False
                        on_lbl = chr(ord(on_lbl)+1)
                else:
                    if onshore_flag == False:
                        off_lbl = chr(ord(off_lbl)+1)
                        onshore_flag = True
                        onshore[ii] = on_lbl
                    else:
                        offshore_flag = False
                        onshore[ii] = on_lbl
                                            
            '''
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
            '''
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
                    if offshore_time >= 0.0 and onshore_time >= 2.0:
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
        if len(station.dwpt.sel(datetime=slice(self.start,self.end)).dropna(dim='datetime',how='all')) > 1:
            dwpt = station.dwpt.sel(datetime=slice(self.start,self.end)).dropna(
                                    dim='datetime',how='all').resample(datetime=sample_rate).interpolate('linear')
            try:
                dwpt_len = len(np.squeeze(dwpt.data))
            except:
                dwpt_len = len(dwpt.data)

            if dwpt_len > 1:
                try:
                    dwpt_before = np.nanmin(dwpt.sel(datetime=slice(self.passage - pd.Timedelta(1.0,'h'),
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
                temp_before = temp.sel(datetime=slice(self.passage - pd.Timedelta(sample_rate),
                                                      self.passage)).data
                change_before = temp_before[1] - temp_before[0]
            except:
                change_before = -999.9
            try:
                temp_after  = temp.sel(datetime=slice(self.passage,self.passage + pd.Timedelta(sample_rate))).data
                change_after  = temp_after[1] - temp_after[0]
            except:
                change_after = 999.9
            if change_before - change_after >= 0.0:
                self.temp_change = True
            else:
                self.temp_change = False
            if verbose: print('Leveling or drop in dry bulb: {}'.format(self.temp_change))
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
        wspd_before = np.nanmin(wspd.sel(datetime=slice(self.passage - pd.Timedelta(1.0,'h'),self.passage)).data)
        wspd_after  = wspd.sel(datetime=slice(self.passage,self.passage + pd.Timedelta(1.0,'h')))
        wspd_gust   = np.max(wspd.sel(datetime=slice(self.passage,self.passage + pd.Timedelta(1.0,'h'))).data)
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

            tot_rain = (rainc - rainc[0]) + (rainnc - rainnc[0])
            #tot_rain = (rainc[1:] - rainc[:-1]) + (rainnc[1:] - rainnc[:-1])
            #tot_rain = np.concatenate([np.asarray([0.0]),tot_rain])
            tot_rain[np.where(tot_rain < 0.25)] = tot_rain[np.where(tot_rain < 0.25)]*0.0
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
        cloud_val = {'nan': 0.0, 'CLR': 0.0, 'FEW': 1.5, 'SCT': 3.5, 'BKN': 6.0, 'OVC': 8.0}
        for ctype in ctype_count.keys():
            if ctype != '' and ctype != ' ' and ctype !='A7:': #and ctype != 'nan' :
                cloud_sum += ctype_count[ctype]*cloud_val[ctype]
                nobs += ctype_count[ctype]
            if verbose: print(ctype,ctype_count[ctype])
        if nobs == 0: 
            cloudy_skies = False
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
        #wdir = wdir.sel(datetime=slice(self.passage,self.passage+pd.Timedelta(1.5,'h')))
        wdir = wdir.sel(datetime=slice(self.passage,self.end))
        
        is_offshore = (wdir>station.onshore_max) | (wdir<station.onshore_min) & (wdir <= 360.0)
        is_onshore  = ~is_offshore
        max_off = 0.0
        off_flag = False
        for off_val in is_offshore:
            if off_val.data:
                if off_flag == False:
                    off_flag = True
                    off_start = off_val.datetime
                    off_end = off_val.datetime
                else:
                    off_end = off_val.datetime
                    time_off = (off_end.data - off_start.data)/np.timedelta64(1,'h')
                    if time_off > max_off: max_off = time_off
            else:
                if off_flag:
                    off_end = off_val.datetime
                    time_off = (off_end.data - off_start.data)/np.timedelta64(1,'h')
                    if time_off > max_off: max_off = time_off
                    off_flag = False
        if verbose: print('max_off',max_off)
        any_onshore = np.any(is_onshore.data)
        
        if method == 'StaufferThompson2015':
            inland_winds_with_onshore = np.squeeze(wspd.data)[np.where(np.squeeze(is_onshore.data))]
            weak_winds = np.all(inland_winds_with_onshore <= light_winds)
            any_onshore = max_off < 2.0
            if (not any_onshore) or weak_winds: 
                self.onshore_inland = False
            else:
                self.onshore_inland = True
        else:
            weak_winds = 'N/A'
            onshore_count = 0
            if any_onshore:
                for vv in np.squeeze(is_onshore.data):
                    if vv: onshore_count += 1
                if 100.0*(onshore_count/len(np.squeeze(is_onshore.data))) < 25.0:
                    self.onshore_inland = False
                else:
                    self.onshore_inland = True
            else:
                self.onshore_inland = False
            #self.onshore_inland = any_onshore

        onshore_count = 0
        if any_onshore:
            for vv in np.squeeze(is_onshore.data):
                if vv: onshore_count += 1
            
        if verbose:
            print('Inland wind direction is onshore: {}'.format(any_onshore))
            print('Inland wind speed is weak: {}'.format(weak_winds))
            if np.any(is_onshore.data):
                print('Number of onshore measurements: {0} out of {1} ({2:2.2f}%)'.format(
                       onshore_count,len(np.squeeze(is_onshore.data)),
                       100.0*(onshore_count/len(np.squeeze(is_onshore.data)))))
            


            
def convert_met_to_math(met_deg):
    math_deg = 90.0 - met_deg
    if math_deg < 0.0: math_deg+=360.0
    return math_deg


def find_onshore_minmax(land_mask,
                        dx = 3.0,
                        dy = 3.0,
                        max_water_dist = 80.0,
                        low_pct_0 = 95.0,
                        upr_pct_0 = 5.0,
                        max_deg_range = 180.0,
                        min_water_size=8,
                        test_mode=False,
                        i_s=None,i_e=None,j_s=None,j_e=None):

    water_mask = land_mask.copy().where(land_mask==0.0) + 1.0
    nx,ny = np.shape(land_mask)
    x = np.arange(0,nx)*dx
    y = np.arange(0,nx)*dy
    xy,yx = np.meshgrid(x,y)
    
    window_len = int(max_water_dist/dx)*2
    half_window_len = int(window_len/2)
    window_center = int((window_len)/2)
    window_dist = ((xy[:window_len+1,:window_len+1] - xy[window_center,window_center])**2 + 
                   (yx[:window_len+1,:window_len+1] - yx[window_center,window_center])**2)**0.5
    window_dist[np.where(window_dist > max_water_dist)] = np.nan
    window_filter = window_dist / window_dist

    window_x,window_y = np.meshgrid(np.arange(0,np.shape(window_dist)[1]+1)*dy - max_water_dist - 1.5,
                                    np.arange(0,np.shape(window_dist)[0]+1)*dx - max_water_dist - 1.5)

    window_deg = -1*(180.0*np.arctan(((yx[:window_len+1,:window_len+1] - yx[window_center,window_center])/
                         (xy[:window_len+1,:window_len+1] - xy[window_center,window_center])))/(np.pi) - 90.0)
    window_deg[:,:half_window_len] = window_deg[:,:half_window_len] + 180.0
    window_deg[np.where(np.isnan(window_dist))] = np.nan

    math_deg = 180.0*np.arctan(((yx[:window_len+1,:window_len+1] - yx[window_center,window_center])/
                         (xy[:window_len+1,:window_len+1] - xy[window_center,window_center])))/(np.pi)
    math_deg[:,:half_window_len] = math_deg[:,:half_window_len] + 180.0
    math_deg[:half_window_len,half_window_len:] = 360 + math_deg[:half_window_len,half_window_len:]
    math_deg[np.where(np.isnan(window_dist))] = np.nan

    '''
    plt.figure(figsize=(14,5))
    plt.subplot(121,aspect='equal')
    plt.pcolormesh(window_x,window_y,window_dist,cmap=plt.cm.viridis)
    plt.colorbar()
    plt.tick_params(labelsize=15)
    plt.ylabel('Distance [km]',size=16)
    plt.xlabel('Distance [km]',size=16)
    plt.subplot(122,aspect='equal')
    plt.pcolormesh(window_x,window_y,window_deg,cmap=plt.cm.viridis)
    plt.colorbar()
    plt.tick_params(labelsize=15)
    plt.xlabel('Distance [km]',size=16)
    plt.show()
    '''
    onshore_min = np.zeros((ny,nx))*np.nan
    onshore_max = np.zeros((ny,nx))*np.nan
    if not test_mode:
        i_s,i_e = half_window_len,nx-half_window_len
        j_s,j_e = half_window_len,ny-half_window_len

    for ii in np.arange(i_s,i_e):
        for jj in np.arange(j_s,j_e): 
            if land_mask[jj,ii] == 1.0:
                loc_water_mask = water_mask[jj-half_window_len:jj+half_window_len+1, ii-half_window_len:ii+half_window_len+1]
                dist_water = loc_water_mask * window_dist
                deg_water  = loc_water_mask * window_deg

                # Break down the water bodies into groups:
                water_bodies = skimage.morphology.label(~np.isnan(deg_water)).astype(np.float32)
                water_bodies[water_bodies==0.0] = np.nan
                water_bodies_orig = water_bodies.copy()

                water_body_size = {}
                water_body_dist = {}
                min_water_distance = 999.9
                closest_water_body = 0.0

                # If we have water bodies to check, enter loop:
                if ~np.all(np.isnan(water_bodies)): 
                    for i in np.arange(1.0,np.nanmax(water_bodies)+1.0): # Loop over all identified water bodies
                        water_size = len(water_bodies[water_bodies==i])
                        if water_size < min_water_size: # Only check for large bodies
                            water_bodies[water_bodies==i] = np.nan
                        else:
                            water_body_size[i] = water_size
                            water_body = water_bodies.copy()
                            water_body[water_bodies!=i] = np.nan # Check only this water body
                            water_body[~np.isnan(water_body)] = 1.0 # Set values to 1 to get mask for distance calculation
                            water_body_min_dist = np.nanpercentile(water_body*dist_water,50)
                            water_body_dist[i] = water_body_min_dist

                    # Small water bodies were removed, check to see if there are any large ones:
                    if ~np.all(np.isnan(water_bodies)):
                        # Find the largest and closest water bodies:
                        largest_water_body = max(water_body_size,key=water_body_size.get)
                        water_body_id = largest_water_body
                        # Loop over all other water bodies
                        for i in water_body_size.keys():
                            
                            deg_mask = water_bodies.copy()
                            deg_mask[water_bodies!=i] = np.nan
                            deg_mask[~np.isnan(deg_mask)] = 1.0
                            
                            deg_water_body = deg_water.where(deg_mask==1.0)
                            deg_range = float(np.nanmax(deg_water_body)) - float(np.nanmin(deg_water_body))
                            
                            # ORIGINAL:
                            #if deg_range > 300:
                            #    deg_water_body[np.where(deg_water_body>300)] -= 360.0
                            if (deg_range > 300) & (deg_range <= 350):
                                if np.nanpercentile(deg_water_body,50) > 260:
                                    deg_water_body[np.where(deg_water_body<15)] += 360
                                elif np.nanpercentile(deg_water_body,50) < 105:
                                    deg_water_body[np.where(deg_water_body>345)] -= 360
                                    
                            if deg_range > 350:
                                if np.nanpercentile(deg_water_body,50) > 225:
                                    deg_water_body[np.where(deg_water_body<15)] += 360
                                elif np.nanpercentile(deg_water_body,50) < 135:
                                    deg_water_body[np.where(deg_water_body>345)] -= 360
                                else:
                                    high_nums = np.count_nonzero(deg_water_body>210)
                                    low_nums  = np.count_nonzero(deg_water_body<150)
                                    if high_nums > low_nums:
                                        deg_water_body[np.where(deg_water_body<35)] += 360
                                    else:
                                        deg_water_body[np.where(deg_water_body>325)] -= 360
                                    


                            #plt.pcolormesh(deg_water_body)
                            #plt.colorbar()
                            #plt.show()
                            
                            deg_water[np.where(deg_mask==1.0)] = deg_water_body[np.where(deg_mask==1.0)]
                            
                            

                            if i != water_body_id:
                                # Check to see if this water body is still relatively large:
                                if water_body_size[i] >= 0.5*water_body_size[water_body_id]:
                                    # Assign this the same water body ID
                                    water_bodies[water_bodies==i] = water_body_id
                                else:
                                    water_bodies[water_bodies==i] = np.nan
                                    
                                    
                        #plt.pcolormesh(deg_water)
                        #plt.colorbar()
                        #plt.show()
                        #wefwef

                        # Set the selected water body (bodies) to 1.0 for masking
                        water_bodies[water_bodies==water_body_id] = 1.0

                        # Multiply water body mask by direction to water:
                        deg_water *= water_bodies


                        # Check to see if there are negative and positive values in the same water body:
                        deg_range = float(np.nanmax(deg_water)) - float(np.nanmin(deg_water))
                        #plt.pcolormesh(deg_water)
                        #plt.colorbar()
                        #plt.show()
                        
                        # ORIGINAL:
                        #if deg_range > 300:
                        #    deg_water[np.where(deg_water>300)] -= 360.0

                        # Set limits for upper and lower bounds.
                        # If range is too big (> max_deg_range) then we iterate by 5 degrees
                        # ... on the upper and lower limits until the range is sufficient.
                        if np.nanmax(water_bodies) > 0:
                            good_lims = False
                            low_pct = low_pct_0
                            upr_pct = upr_pct_0
                            while good_lims == False:
                                lowr_lim = np.nanpercentile(deg_water,low_pct)
                                uppr_lim = np.nanpercentile(deg_water,upr_pct)
                                if lowr_lim - uppr_lim < max_deg_range:
                                    good_lims = True
                                else:
                                    low_pct -= 5.0
                                    upr_pct += 5.0                                
                        else: # Set limits to nan when water_bodies is all nan
                            lowr_lim = np.nan
                            uppr_lim = np.nan

                        onshore_min[jj,ii] = uppr_lim 
                        onshore_max[jj,ii] = lowr_lim


                    if test_mode:
                        lwr_xe = max_water_dist*np.cos(np.radians(convert_met_to_math(lowr_lim)))
                        lwr_ye = max_water_dist*np.sin(np.radians(convert_met_to_math(lowr_lim)))
                        upr_xe = max_water_dist*np.cos(np.radians(convert_met_to_math(uppr_lim)))
                        upr_ye = max_water_dist*np.sin(np.radians(convert_met_to_math(uppr_lim)))

                        mid_xe = max_water_dist*np.cos(np.radians(convert_met_to_math(np.nanmedian(deg_water))))
                        mid_ye = max_water_dist*np.sin(np.radians(convert_met_to_math(np.nanmedian(deg_water))))

                        
                        
                        fig = plt.figure(figsize=(18,9))
                        lm_pltF = plt.subplot2grid((2,3),(0,0),colspan=2,rowspan=2,aspect='equal')
                        dist_plt = plt.subplot2grid((2,3),(0,2),aspect='equal')
                        deg_plt = plt.subplot2grid((2,3),(1,2),aspect='equal')
                        lon = land_mask.XLONG
                        lat = land_mask.XLAT
                        plt_landmask = lm_pltF.pcolormesh(xy,yx,land_mask.where(land_mask==0.0),
                                                          cmap=plt.cm.Greys_r,
                                                          rasterized=True)#,levels=[0.5],colors='k')
                        lm_pltF.scatter(xy[jj,ii],yx[jj,ii],facecolor='b',marker='*',s=200)
                        max_water_dist_ind = int(np.ceil(3000.0/max_water_dist))
                        circ_latlon_dist = lon[jj+max_water_dist_ind,ii]
                        draw_circle = plt.Circle((xy[jj,ii],yx[jj,ii]), max_water_dist,fill=False)
                        lm_pltF.add_artist(draw_circle)
                        lm_pltF.tick_params(labelsize=15)
                        lm_pltF.set_xlabel('Distance [km]',size=18)
                        lm_pltF.set_ylabel('Distance [km]',size=18)
                        lm_pltF.set_title('Location of Interest',size=18)
                        lm_pltF.text(10,590,'a.)',size=18,ha='left',va='top')

                        dist_plt_cm = dist_plt.pcolormesh(window_x,window_y,water_bodies_orig,
                                                          #norm=Normalize(0,max_water_dist),
                                                          cmap=plt.cm.tab20,
                                                          rasterized=True)
                        dist_cbar = plt.colorbar(dist_plt_cm,ax=dist_plt)
                        dist_cbar.ax.tick_params(labelsize=14)
                        dist_cbar.ax.set_title('[km]')
                        dist_plt.scatter(0,0,facecolor='b',marker='*',s=400)
                        dist_plt.set_xlim(np.nanmin(window_x)-5.0,np.nanmax(window_x)+5.0)
                        dist_plt.set_ylim(np.nanmin(window_y)-5.0,np.nanmax(window_y)+5.0)
                        dist_plt.tick_params(labelsize=15,labelbottom=False)
                        draw_circle = plt.Circle((0.0, 0.0), max_water_dist,fill=False)
                        dist_plt.add_artist(draw_circle)
                        #dist_plt.set_ylabel('Distance [km]',size=18)
                        dist_plt.set_title('Water ID',size=18)
                        dist_plt.text(-110,105,'b.)',size=18,ha='left',va='top')
                        dist_plt.text(40,-90,'I',size=17,color='lime')
                        dist_plt.text(-50,-35,'III',size=17,color='r')
                        dist_plt.text(22,100,'IV',size=17,color='r',va='top',ha='center')
                        dist_plt.text(-10,-55,'II',size=17,color='r',va='center',ha='center')

                        deg_plt_cm = deg_plt.pcolormesh(window_x,window_y,deg_water,cmap=plt.cm.hsv,
                                                        norm=Normalize(0,360),rasterized=True)
                        deg_cbar = plt.colorbar(deg_plt_cm,ax=deg_plt)
                        deg_cbar.set_ticks(np.arange(0,361,90))
                        deg_cbar.ax.tick_params(labelsize=14)
                        deg_cbar.ax.set_title('[˚]')
                        lim_color = 'darkgreen'
                        deg_plt.plot([0,lwr_xe],[0,lwr_ye],c=lim_color)
                        lwr_pct_str = '{}'.format(int(low_pct))
                        upr_pct_str = '{}'.format(int(upr_pct))
                        deg_plt.text(lwr_xe*0.5,lwr_ye*0.35,lwr_pct_str + '$^{\mathrm{th}}$',size=15,ha='right',color=lim_color)
                        deg_plt.plot([0,upr_xe],[0,upr_ye],c=lim_color)
                        deg_plt.text(upr_xe*1.15,upr_ye*1.15,upr_pct_str+'$^{\mathrm{th}}$',size=15,ha='center',color=lim_color,va='center')
                        deg_plt.plot([0,mid_xe],[0,mid_ye],c='k',ls=':')
                        deg_plt.scatter(0,0,facecolor='b',marker='*',s=400,zorder=5)
                        deg_plt.set_xlim(np.nanmin(window_x)-5.0,np.nanmax(window_x)+5.0)
                        deg_plt.set_ylim(np.nanmin(window_y)-5.0,np.nanmax(window_y)+5.0)
                        deg_plt.tick_params(labelsize=15)
                        deg_plt.set_ylabel('Distance [km]',size=18)
                        deg_plt.set_xlabel('Distance [km]',size=18)
                        deg_plt.set_title('Water Direction',size=18)
                        deg_plt.text(-110,105,'c.)',size=18,ha='left',va='top')
                        pts  = np.arange(uppr_lim,lowr_lim,2.0)
                        fill_c = 'goldenrod'
                        x, y = 0,0
                        npts = pts.size
                        fill_x = [x]
                        fill_y = [y]
                        for dd,wdir in enumerate(pts):
                            d = 270.0 - wdir    # Convert met degrees to polar
                            plt_dist = -max_water_dist # Met degrees are FROM dir... need negative distance!
                            fill_x.append(x+plt_dist*np.cos(np.radians(d)))
                            fill_y.append(y+plt_dist*np.sin(np.radians(d)))

                        deg_plt.fill(fill_x, fill_y,alpha=0.10,lw=None,color=fill_c,zorder=2)
                        draw_circle = plt.Circle((0.0, 0.0), max_water_dist,fill=False)
                        deg_plt.add_artist(draw_circle)
                        deg_plt.text(40,-90,'I',size=17,color='k')
                        deg_plt.text(-50,-35,'III',size=17,color='r',alpha=0.4)
                        deg_plt.text(22,100,'IV',size=17,color='r',va='top',ha='center',alpha=0.4)
                        deg_plt.text(-10,-55,'II',size=17,color='r',va='center',ha='center',alpha=0.4)
                        return(fig)
                        #plt.show()
                        
                        print()
                        #wefwef
    
    onshore_min_da = land_mask.copy()
    onshore_min_da.data = onshore_min
    onshore_max_da = land_mask.copy()
    onshore_max_da.data = onshore_max
    onshore_min_max_ds = xr.Dataset({'onshore_min':onshore_min_da,
                                     'onshore_max':onshore_max_da})
    onshore_min_max_ds.attrs['dx'] = dx
    onshore_min_max_ds.attrs['dy'] = dy
    onshore_min_max_ds.attrs['max_water_dist'] = max_water_dist
    onshore_min_max_ds.attrs['max_deg_range'] = max_deg_range
    
    return(onshore_min_max_ds)
