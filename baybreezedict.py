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



class BayBreezeDetection():
    '''
    Detect bay breeze events with several different methods. Validate the events
    with inland station data.

    Methods implemented: "StaufferThompson2015"
                         "Stauffer2015"
                         "Sikora2010" (not yet available...)
    '''
    def __init__(self,station,inland_stations,resample=False,sample_rate='60min',light_winds=3.08667,show_plot=False,
                 method='StaufferThompson2015',min_points=3):
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
        '''
        bay_breeze_detected  = False
        bay_breeze_validated = False

        if station.wspd.dropna(dim='datetime',how='any').size >= min_points & \
           station.wdir.dropna(dim='datetime',how='any').size >= min_points:
            self._detect_wind_shift(station,resample=resample)
            if self.wind_shift:
                var_names = []
                for dd in station.data_vars: var_names.append(dd)
                if 'cldc' in var_names:
                    station_cld = station.cldc.dropna(dim='datetime',how='any')
                elif ('skyc1' in var_names) and ('skyc2' in var_names) and ('skyc3' in var_names) and ('skyc4' in var_names):
                    skyc1 = station.skyc1.values#.dropna(dim='datetime',how='any')
                    station_cld = ['']*len(skyc1)

                    for vv,val in enumerate(skyc1):
                        station_cld[vv] = '{},{},{},{}'.format(str(val).strip(),str(station.skyc2.values[vv]).strip(),
                                                        str(station.skyc3.values[vv]).strip(),str(station.skyc4.values[vv]).strip())
                        if 'nan' in station_cld[vv]: station_cld[vv] = 'nan'
                        if station_cld[vv] == ',,,': station_cld[vv] = 'nan'
                        if station_cld[vv] == 'VV,,,': station_cld[vv] = 'nan'
                    station_cld = xr.DataArray(station_cld,dims=['datetime'],coords=[station.datetime.values])
            
                station_pcp = station.pcip.dropna(dim='datetime',how='any')
                if len(station_cld.dropna(dim='datetime',how='any')) > 0:
                    print('evaluating clouds')
                    self._check_clouds_and_precip(station_cld,station_pcp)
                if self.clear_and_dry:
                    if method=='Stauffer2015':
                        self._check_1degDwptRise(station,resample=resample,show_plot=show_plot)
                        if self.dwpt_rise:
                            print('BAY BREEZE DETECTED')
                            bay_breeze_detected = True
                    else:
                        print('BAY BREEZE DETECTED')
                        bay_breeze_detected = True

        self.detected  = bay_breeze_detected
        self.validated = bay_breeze_validated

        if self.detected:
            self._validate_bay_breeze(station,inland_stations=inland_stations,resample=resample,show_plot=show_plot)

    def _check_1degDwptRise(self,station,resample=True,show_plot=False):
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
            print(station_dpt.values)
            print(dpt_before.values,dpt_after.values)
            if len(dpt_before.values) > 0 and len(dpt_after.values) > 0:
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
        


    def _validate_bay_breeze(self, station, inland_stations, light_winds=3.08667, resample=True, show_plot=False):
        station_names = inland_stations['station'].values
        n_inland = len(station_names)
        station_validated = [False]*n_inland
        bay_breeze_validated  = False
        low_winds_in_period   = False
        onshore_dir_validated = False
        if show_plot: fig,ax = plt.subplots(nrows=2,sharex=True)
        for sin,inland in enumerate(station_names):
            if resample:
                inland_wspd = inland_stations.sel(station=inland).wspd.dropna(dim='datetime',how='any').resample(
                                                                datetime=sample_rate).interpolate('linear')
                inland_wdir = inland_stations.sel(station=inland).wdir.dropna(dim='datetime',how='any').resample(
                                                                datetime=sample_rate).interpolate('linear')
            else:
                inland_wspd = inland_stations.sel(station=inland).wspd.dropna(dim='datetime',how='any')
                inland_wdir = inland_stations.sel(station=inland).wdir.dropna(dim='datetime',how='any') 
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
                inland_wspd.plot.line(marker='o',ax=ax[0],label=inland)
                inland_wdir.plot.line(marker='o',ax=ax[1],label=inland)
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
            if ctype != '' and ctype != 'nan':
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
        clear_and_dry = clear_skies & dry
        self.clear_and_dry = clear_and_dry


    def _detect_wind_shift(self, station, sample_rate='60min', light_winds=3.08667, resample=True):
        '''
        From Stauffer and Thompson, 2015:

        For each day, the daytime (0900 to 1600 Eastern Standard Time, EST) wind directions were evaluated: 
            
        If the hourly wind direction measurement changed from either offshore, calm, or light and variable (less than 6 kt), 
        to onshore sustained for two or more consecutive hours during the period...
        '''

        _, index = np.unique(station['datetime'], return_index=True)
        station = station.isel(datetime=index)
        if resample:
            station_spd = station.wspd.dropna(dim='datetime',how='any').resample(
                                            datetime=sample_rate).interpolate('linear')
            station_dir = station.wdir.dropna(dim='datetime',how='any').resample(
                                            datetime=sample_rate).interpolate('linear')
            station_tmp = station.temp.dropna(dim='datetime',how='any').resample(
                                            datetime=sample_rate).interpolate('linear')
        else:
            station_spd = station.wspd.dropna(dim='datetime',how='any')
            station_dir = station.wdir.dropna(dim='datetime',how='any')

        is_onshore  = ((station_dir>=station.onshore_min) & (station_dir <= station.onshore_max)
                        & (station_dir <= 360.0)).data.squeeze()
        is_offshore = ((station_dir<station.onshore_min) | \
                      ((station_dir>station.onshore_max) & (station_dir<=360.0))).data.squeeze()
        is_lt_vrb   = ((station_spd<light_winds) & (station_dir > 360.0)).data.squeeze()
        if len(is_lt_vrb) == len(is_offshore):
            offshore_conditions = is_lt_vrb | is_offshore
        else:
            offshore_conditions = False
        #print('   offshore: {}'.format(is_offshore))
        #print('light & vrb: {}'.format(is_lt_vrb))
        #print(' eaither or: {}'.format(offshore_conditions))
        #print('    onshore: {}'.format(is_onshore))
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
                        #off_lbl = chr(ord(off_lbl)+1)
                        #new_on_lbl = False

            #    print(off_lbl,on_lbl)
            offshore_count = Counter(offshore) 
            onshore_count  = Counter(onshore)
            for lbl in offshore_count.keys():
                if len(lbl) > 0:
                    if lbl in onshore_count.keys():
                        if resample:
                            offshore_time = (offshore_count[lbl]-1)*pd.to_timedelta(sample_rate) / np.timedelta64(1, 'h')
                            onshore_time  = (onshore_count[lbl]-1)*pd.to_timedelta(sample_rate) / np.timedelta64(1, 'h')
                        else:
                            offshore_inds = np.where(np.asarray(offshore)==lbl)[0]
                            offshore_s    = offshore_inds[0]
                            offshore_e    = offshore_inds[-1]

                            onshore_inds  = np.where(np.asarray(onshore)==lbl)[0]
                            onshore_s     = onshore_inds[0]
                            onshore_e     = onshore_inds[-1]

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
                        #print('Label {} offshore: {} hours'.format(lbl,offshore_time))
                        #print('Label {} onshore:  {} hours'.format(lbl,onshore_time))
                        if offshore_time >= 1.0 and onshore_time >= 2.0:
                            bay_breeze_start = station_spd.datetime[np.where(np.asarray(offshore)==lbl)[0][0]]
                            bay_breeze_pass  = station_spd.datetime[np.where(np.asarray(onshore)==lbl)[0][0]]
                            bay_breeze_end   = station_spd.datetime[np.where(np.asarray(onshore)==lbl)[0][-1]]
                            #print(bay_breeze_start.data)
                            #print(bay_breeze_pass.data)
                            #print(bay_breeze_end.data)
                            if bay_breeze_start < bay_breeze_pass:
                                wind_shift = True

        self.wind_shift = wind_shift
        self.start    = bay_breeze_start
        self.passage  = bay_breeze_pass 
        self.end      = bay_breeze_end  
#        return(bay_breeze_detected,bay_breeze_start,bay_breeze_pass,bay_breeze_end)
