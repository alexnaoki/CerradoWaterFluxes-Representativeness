import pandas as pd
import numpy as np
import pathlib
import matplotlib.pyplot as plt
import datetime as dt
# import tensorflow as tf
import calendar
import math
import matplotlib.dates as mdates
import seaborn as sns

class InitializeAnalysis:
    def __init__(self, ep_path, lf_path, iab2_path, iab1_path, footprint_file, k15_path):
        # lai_file_path, evi_file_path
        # File's Path
        self.iab3EP_path = pathlib.Path(ep_path)
        self.iab3LF_path = pathlib.Path(lf_path)
        self.iab2_path = pathlib.Path(iab2_path)
        self.iab1_path = pathlib.Path(iab1_path)

        self.footprint_file = pathlib.Path(footprint_file)
        self.k15_path = pathlib.Path(k15_path)
        # self.lai_path = pathlib.Path(lai_file_path)
        # self.evi_path = pathlib.Path(evi_file_path)


        self._read_file()

    def _read_file(self):
        # Reading csv files
        self.iab3EP_files = self.iab3EP_path.rglob('eddypro*p08*full*.csv')
        self.iab3LF_files = self.iab3LF_path.rglob('TOA5*.flux*.dat')
        self.iab2_files = self.iab2_path.rglob('*.dat')
        self.iab1_files = self.iab1_path.rglob('*Table1*.dat')
        self.k15_files = self.k15_path.rglob('*n.csv')

        footprint_columns = ['TIMESTAMP','code03', 'code04', 'code09','code12','code15','code19','code20','code24','code25','code33']
        ep_columns =  ['date','time',  'H', 'qc_H', 'LE', 'qc_LE','sonic_temperature', 'air_temperature', 'air_pressure', 'air_density',
               'ET', 'e', 'es', 'RH', 'VPD','Tdew', 'u_unrot', 'v_unrot', 'w_unrot', 'u_rot', 'v_rot', 'w_rot', 'wind_speed',
               'max_wind_speed', 'wind_dir', 'u*', '(z-d)/L',  'un_H', 'H_scf', 'un_LE', 'LE_scf','u_var', 'v_var', 'w_var', 'ts_var','H_strg','LE_strg', 'co2_flux']
        lf_columns = ['TIMESTAMP', 'CO2_sig_strgth_mean','H2O_sig_strgth_mean','Rn_Avg','Rs_incoming_Avg', 'Rs_outgoing_Avg',
                      'Rl_incoming_Avg', 'Rl_outgoing_Avg', 'Rl_incoming_meas_Avg','Rl_outgoing_meas_Avg', 'shf_Avg(1)', 'shf_Avg(2)',
                      'precip_Tot']
        iab3EP_dfs = []
        print('Reading IAB3_EP files...')
        for file in self.iab3EP_files:
            iab3EP_dfs.append(pd.read_csv(file,skiprows=[0,2], na_values=-9999, parse_dates={'TIMESTAMP':['date','time']}, keep_date_col=True, usecols=ep_columns))
        self.iab3EP_df = pd.concat(iab3EP_dfs)
        print(f"# IAB3_EP: {len(iab3EP_dfs)}\tInicio: {self.iab3EP_df['TIMESTAMP'].min()}\tFim: {self.iab3EP_df['TIMESTAMP'].max()}")

        iab3LF_dfs = []
        print('Reading IAB3_LF files...')
        for file in self.iab3LF_files:
            iab3LF_dfs.append(pd.read_csv(file, skiprows=[0,2,3], na_values=['NAN'], parse_dates=['TIMESTAMP'], usecols=lf_columns))
        self.iab3LF_df = pd.concat(iab3LF_dfs)
        print(f"# IAB3_LF: {len(iab3LF_dfs)}\tInicio:{self.iab3LF_df['TIMESTAMP'].min()}\tFim: {self.iab3LF_df['TIMESTAMP'].max()}")

        iab2_dfs = []
        print('Reading IAB2 files...')
        for file in self.iab2_files:
            iab2_dfs.append(pd.read_csv(file, skiprows=[0,2,3], na_values=['NAN'], parse_dates=['TIMESTAMP']))
        self.iab2_df = pd.concat(iab2_dfs)
        print(f"# IAB2: {len(iab2_dfs)}\tInicio: {self.iab2_df['TIMESTAMP'].min()}\tFim: {self.iab2_df['TIMESTAMP'].max()}")

        iab1_dfs = []
        print('Reading IAB1 files...')
        for file in self.iab1_files:
            iab1_dfs.append(pd.read_csv(file, skiprows=[0,2,3], na_values=['NAN'], parse_dates=['TIMESTAMP']))
        self.iab1_df = pd.concat(iab1_dfs)
        print(f"# IAB1: {len(iab1_dfs)}\tInicio: {self.iab1_df['TIMESTAMP'].min()}\tFim: {self.iab1_df['TIMESTAMP'].max()}")

        iab_dfs = [self.iab3EP_df, self.iab3LF_df, self.iab2_df, self.iab1_df]

        print('Reading Footprint file...')
        self.footprint_df = pd.read_csv(self.footprint_file, parse_dates=['TIMESTAMP'], na_values=-9999, usecols=footprint_columns)
        self.footprint_df.drop_duplicates(subset='TIMESTAMP', inplace=True)
        print(f"Inicio: {self.footprint_df['TIMESTAMP'].min()}\tFim: {self.footprint_df['TIMESTAMP'].max()}")

        print('Reading K15 files...')
        k15_files = []
        for file in self.k15_files:
            k15_files.append(pd.read_csv(file, parse_dates=['datetime']))
        self.k15_df = pd.concat(k15_files)
        print(f"# K15: {len(k15_files)}\tInicio: {self.k15_df['datetime'].min()}\tFim: {self.k15_df['datetime'].max()}")

        # print('Reading LAI file...')
        # self.lai_df = pd.read_csv(self.lai_path, parse_dates=['datetime'])

        # print('Reading EVI file...')
        # self.evi_df = pd.read_csv(self.evi_path, parse_dates=['datetime'])

        # Removing duplicated files based on 'TIMESTAMP'
        for df in iab_dfs:
            print('Duplicatas: ',df.duplicated().sum())
            df.drop_duplicates(subset='TIMESTAMP', keep='first', inplace=True)
            df.reset_index(inplace=True)
            print('Verificacao de Duplicatas: ', df.duplicated().sum())


        # Merging files from EddyPro data and LowFreq data
        self.iab3_df = pd.merge(left=self.iab3EP_df, right=self.iab3LF_df, on='TIMESTAMP', how='inner')

        # Merging EP and LF data with footprint data
        self.iab3_df = pd.merge(left=self.iab3_df, right=self.footprint_df, on='TIMESTAMP', how='inner')

        # Merging EP and LF data with K15 data
        self.iab3_df = pd.merge(left=self.iab3_df, right=self.k15_df, left_on='TIMESTAMP', right_on='datetime', how='inner')

        # Merging EP and LF data with LAI data
        # self.iab3_df = pd.merge(left=self.iab3_df, right=self.lai_df, left_on='TIMESTAMP', right_on='datetime', how='inner')

        # Merging EP and LF data with EVI data
        # self.iab3_df = pd.merge(left=self.iab3_df, right=self.evi_df, left_on='TIMESTAMP', right_on='datetime', how='inner')

        # print(self.iab3_df.loc[(self.iab3_df['TIMESTAMP'].dt.year==2020)&(self.iab3_df['TIMESTAMP'].dt.month==11),'ET'].describe())
        # print(self.iab3EP_df.loc[(self.iab3EP_df['TIMESTAMP'].dt.year==2020)&(self.iab3EP_df['TIMESTAMP'].dt.month==11),'ET'].describe())
        # print(self.iab3LF_df.loc[(self.iab3LF_df['TIMESTAMP'].dt.year==2020)&(self.iab3LF_df['TIMESTAMP'].dt.month==11)].describe())


        # Resampling IAB2
        self.iab2_df_resample = self.iab2_df.set_index('TIMESTAMP').resample('30min').mean()
        self.iab2_df_resample.reset_index(inplace=True)
        # print(self.iab2_df_resample)

    def _applying_basic_filters(self):
        # Flag using Mauder and Foken (2004)
        self.iab3_df.loc[self.iab3_df[['qc_H','qc_LE']].isin([0]).sum(axis=1)==2, 'flag_qaqc'] = 1
        self.iab3_df.loc[self.iab3_df[['qc_H','qc_LE']].isin([0]).sum(axis=1)!=2, 'flag_qaqc'] = 0
        
        # Flag rain
        self.iab3_df.loc[self.iab3_df['precip_Tot']>0, 'flag_rain'] = 0
        self.iab3_df.loc[self.iab3_df['precip_Tot']==0, 'flag_rain'] = 1

        # Flag signal strength
        min_signalStr = 0.8
        self.iab3_df.loc[self.iab3_df['H2O_sig_strgth_mean']>=min_signalStr, 'flag_signalStr'] = 1
        self.iab3_df.loc[self.iab3_df['H2O_sig_strgth_mean']<min_signalStr, 'flag_signalStr'] = 0

        # Flag footprint
        self.iab3_df['footprint_acceptance'] = self.iab3_df[['code03', 'code04']].sum(axis=1)/self.iab3_df[['code03','code04','code09','code12','code15','code19','code20','code24','code25','code33']].sum(axis=1)
        min_footprint = 0.8
        self.iab3_df.loc[self.iab3_df['footprint_acceptance']>=min_footprint, 'flag_footprint'] = 1
        self.iab3_df.loc[self.iab3_df['footprint_acceptance']<min_footprint, 'flag_footprint'] = 0

        # Flag K15
        min_k15 = 0.8
        self.iab3_df.loc[self.iab3_df['k15_filter_n']>=min_k15, 'flag_k15'] = 1
        self.iab3_df.loc[self.iab3_df['k15_filter_n']<min_k15, 'flag_k15'] = 0

        # Flag Basics
        self.iab3_df.loc[self.iab3_df[['flag_qaqc', 'flag_rain', 'flag_signalStr']].isin([1]).sum(axis=1)==3, 'flag_basic'] = 1
        self.iab3_df.loc[self.iab3_df[['flag_qaqc', 'flag_rain', 'flag_signalStr']].isin([1]).sum(axis=1)!=3, 'flag_basic'] = 0

        # Flag Full
        self.iab3_df.loc[self.iab3_df[['flag_qaqc', 'flag_rain', 'flag_signalStr', 'flag_k15']].isin([1]).sum(axis=1)==4, 'flag_full'] = 1
        self.iab3_df.loc[self.iab3_df[['flag_qaqc', 'flag_rain', 'flag_signalStr','flag_k15']].isin([1]).sum(axis=1)!=4, 'flag_full'] = 0

    def _datetime_classification(self):
        self.iab3_df['Hour'] = self.iab3_df['TIMESTAMP'].dt.hour
        self.iab3_df['Day'] = self.iab3_df['TIMESTAMP'].dt.day
        self.iab3_df['Month'] = self.iab3_df['TIMESTAMP'].dt.month

        self.iab3_df.loc[self.iab3_df['TIMESTAMP'].dt.month.isin([1,2,12]), 'Season'] = 'Summer'
        self.iab3_df.loc[self.iab3_df['TIMESTAMP'].dt.month.isin([3,4,5]), 'Season'] = 'Autumn'
        self.iab3_df.loc[self.iab3_df['TIMESTAMP'].dt.month.isin([6,7,8]), 'Season'] = 'Winter'
        self.iab3_df.loc[self.iab3_df['TIMESTAMP'].dt.month.isin([9,10,11]), 'Season'] = 'Spring'

        self.iab3_df.loc[self.iab3_df['TIMESTAMP'].dt.month.isin([1,2,3,10,11,12]), 'Weather'] = 'Rainy'
        self.iab3_df.loc[self.iab3_df['TIMESTAMP'].dt.month.isin([4,5,6,7,8,9]), 'Weather'] = 'Dry'

    def _gagc(self):
        self.iab3_df.loc[(self.iab3_df['flag_qaqc']==0)|
                               (self.iab3_df['flag_rain']==0)|
                               (self.iab3_df['flag_signalStr']==0)
                               |(self.iab3_df['LE']<0)
                               |(self.iab3_df['u*']<=0.1)
                            #    |(self.iab3_df['VPD']==-999900)
                            #    |(self.iab3_df['flag_footprint']==0)
                               , 'LE'] = np.nan

        # self.iab3_df.loc[(self.iab3)]

        self.iab3_df['psychrometric_kPa'] = 0.665*10**(-3)*self.iab3_df['air_pressure']/1000
        self.iab3_df['delta'] = 4098*(0.6108*np.e**(17.27*(self.iab3_df['air_temperature']-273.15)/((self.iab3_df['air_temperature']-273.15)+237.3)))/((self.iab3_df['air_temperature']-273.15)+237.3)**2
        self.iab3_df['VPD_kPa'] = (self.iab3_df['es']-self.iab3_df['e'])/1000
        self.iab3_df['LE_MJmh'] = self.iab3_df['LE']*3600/1000000
        self.iab3_df['Rn_Avg_MJmh'] = self.iab3_df['Rn_Avg']*3600/1000000
        self.iab3_df['shf_Avg_MJmh'] = self.iab3_df[['shf_Avg(1)','shf_Avg(2)']].mean(axis=1)*3600/1000000

        self.iab3_df['ga'] = (self.iab3_df['wind_speed']/self.iab3_df['u*']**2)**(-1)
        self.iab3_df['gc'] = (self.iab3_df['LE_MJmh']*self.iab3_df['psychrometric_kPa']*self.iab3_df['ga'])/(self.iab3_df['delta']*(self.iab3_df['Rn_Avg_MJmh']-self.iab3_df['shf_Avg_MJmh'])+self.iab3_df['air_density']*3600*1.013*10**(-3)*self.iab3_df['VPD_kPa']*self.iab3_df['ga']-self.iab3_df['LE_MJmh']*self.iab3_df['delta']-self.iab3_df['LE_MJmh']*self.iab3_df['psychrometric_kPa'])

        self.iab3_df.loc[self.iab3_df['gc']<0, 'gc'] = np.nan
        self.iab3_df.loc[self.iab3_df['gc']>0.02, 'gc'] = np.nan
        self.iab3_df.loc[self.iab3_df['VPD']==-999900, 'VPD'] = np.nan

        self.iab3_df['ratio_gagc'] = self.iab3_df['ga']/self.iab3_df['gc']
        self.iab3_df['ratio_gcga'] = self.iab3_df['gc']/self.iab3_df['ga']

        self.iab3_df['ratio_deltapsy'] = self.iab3_df['delta']/self.iab3_df['psychrometric_kPa']


    def _decoupling(self):
        # self.iab3_df['Omega'] = (1 + (self.iab3_df['psychrometric_kPa']/(self.iab3_df['psychrometric_kPa']+self.iab3_df['delta']))*(self.iab3_df['ga']/self.iab3_df['gc']))**(-1)
        self.iab3_df['Omega'] = (1+self.iab3_df['delta']/self.iab3_df['psychrometric_kPa'])/((1+self.iab3_df['delta']/self.iab3_df['psychrometric_kPa']+self.iab3_df['ga']/self.iab3_df['gc']))
        self.iab3_df['Omega-1'] = 1/self.iab3_df['Omega']

        self.iab3_df['Omega_max'] = (1 + self.iab3_df['ratio_deltapsy'].max())/(1 + self.iab3_df['ratio_deltapsy'].max() + self.iab3_df['ratio_gagc'])
        self.iab3_df['Omega_min'] = (1 + self.iab3_df['ratio_deltapsy'].min())/(1 + self.iab3_df['ratio_deltapsy'].min() + self.iab3_df['ratio_gagc'])
        self.iab3_df['Omega_med'] = (1 + self.iab3_df['ratio_deltapsy'].mean())/(1 + self.iab3_df['ratio_deltapsy'].mean() + self.iab3_df['ratio_gagc'])



        self.iab3_df.loc[(self.iab3_df['Omega']<0)|(self.iab3_df['Omega']>1), 'Omega'] = np.nan

    def _adjust_wind_direction(self):
        self.iab3_df['wind_dir_sonic'] = 360 - self.iab3_df['wind_dir']
        azimute = 135.1
        # azimute = 45
        self.iab3_df['wind_dir_compass'] = (360 + azimute - self.iab3_df['wind_dir_sonic']).apply(lambda x: x-360 if x>=360 else x)

    def _applying_basic_processing(self):
        self._applying_basic_filters()

        self._datetime_classification()
        self._gagc()
        self._decoupling()
        self._adjust_wind_direction()


    def get_data(self):
        self._applying_basic_processing()
        # self._applying_basic_filters()
        # self._gagc()
        # self._decoupling()
        return self.iab3_df

    def select_datetime(self, year, begin_hour=0, end_hour=23):
        self._applying_basic_processing()

        iab3_selected_datetime = self.iab3_df.loc[(self.iab3_df['TIMESTAMP'].dt.year==year)&
                                              (self.iab3_df['TIMESTAMP'].dt.hour>=begin_hour)&
                                              (self.iab3_df['TIMESTAMP'].dt.hour<=end_hour)]
        iab3_selected_datetime.reset_index()
        return iab3_selected_datetime

    def select_datetime_raw(self, year, begin_hour=0, end_hour=23):
        self._datetime_classification()

        iab3_selected_datetime = self.iab3_df.loc[(self.iab3_df['TIMESTAMP'].dt.year==year)&
                                              (self.iab3_df['TIMESTAMP'].dt.hour>=begin_hour)&
                                              (self.iab3_df['TIMESTAMP'].dt.hour<=end_hour)]
        iab3_selected_datetime.reset_index()
        return iab3_selected_datetime