import sys
sys.path.append(r'C:\Users\User\git\EC-LHC\footprint\FFP_python_v1_4')
from calc_footprint_FFP_adjusted01 import FFP

import pandas as pd
import pathlib
import numpy as np
# from scipy import ndimage
from cupyx.scipy import ndimage
import cupy as cp
from skimage.measure import block_reduce
import matplotlib.pyplot as plt
import rasterio
from rasterio.transform import from_gcps
from rasterio.control import GroundControlPoint as GCP
import xarray
import rioxarray
from rasterio.crs import CRS
import matplotlib.colors as colors
import calendar
import time


class K15_to_raster:
    def __init__(self, path_to_k15_input, path_tif_reference):
        # Tiff reference (MapBiomas)
        self.raster_reference = rasterio.open(path_tif_reference)

        # K15 input
        folder_path_k15_input = pathlib.Path(path_to_k15_input)
        regex_data = '*p08*full_output'
        self.path_k15_input = folder_path_k15_input.rglob(f'{regex_data}*.csv')

        k15_input_columns = ['date','time','wind_dir', 'u_rot','L','v_var','u*','wind_speed']

        dfs_single_config = []
        for file in self.path_k15_input:
            df_single_config = pd.read_csv(file, skiprows=[0,2], na_values=-9999, 
                                           parse_dates={'TIMESTAMP':['date','time']}, 
                                           keep_date_col=True, usecols=k15_input_columns)
            dfs_single_config.append(df_single_config)

        self.df_k15 = pd.concat(dfs_single_config)
        self.df_k15.dropna(inplace=True)
        self.df_k15.sort_values(by='TIMESTAMP', inplace=True)
        self.df_k15.drop_duplicates(subset='TIMESTAMP',inplace=True)

        self.df_k15['wind_dir_sonic'] = 360 - self.df_k15['wind_dir']
        azimute = 135.1
        self.df_k15['wind_dir_compass'] = (360 + azimute - self.df_k15['wind_dir_sonic']).apply(lambda x: x-360 if x>=360 else x)
    
        # IAB3
        self.iab3_x_utm_sirgas = 203917.07880027
        self.iab3_y_utm_sirgas = 7545463.6805863

    def _warp_array(self, k15_output, angle):
        start = time.time()
        x = k15_output[11]
        y = k15_output[12]
        f = k15_output[5]
        
        ### X and Y Bases
        x_array_base = np.linspace(x.min(), x.max(), len(np.unique(x)))
        y_array_base = np.linspace(y.min(), y.max(), len(np.unique(y)))
        Y_array, X_array = np.meshgrid(y_array_base, x_array_base)
        
        dx_base = abs(x_array_base[1] - x_array_base[0])
        dy_base = abs(y_array_base[1] - y_array_base[0])
        
        ### X and Y Pads
        vertical_pad = [x.shape[0]+int(x_array_base.min()//dx_base)+1, 0+int(x_array_base.min()//dx_base+1)]
        horizontal_pad = [y.shape[1]//2, y.shape[1]//2]
        
        x0_base = x_array_base.min() - vertical_pad[0]*dx_base
        y0_base = y_array_base.min() - horizontal_pad[0]*dy_base
        
        x1_base = x_array_base.max() + vertical_pad[1]*dx_base
        y1_base = y_array_base.max() + horizontal_pad[1]*dy_base
        
        X_array_p = np.linspace(x0_base, x1_base, vertical_pad[0]+vertical_pad[1]+len(x_array_base))
        Y_array_p = np.linspace(y0_base, y1_base, horizontal_pad[0]+horizontal_pad[1]+len(y_array_base))
        
        XX_array_p, YY_array_p = np.meshgrid(Y_array_p, X_array_p)
        
        print(time.time()-start)
        ### Pad f and rotate
        start = time.time()

        f_p = np.pad(f, [vertical_pad, horizontal_pad], 'constant')
        f_p = cp.asarray(f_p)
        f_R = ndimage.rotate(f_p, angle, reshape=False)
        f_R = cp.asnumpy(f_R)
        # plt.pcolormesh(f_R)
        # plt.show()
        print('rotate',time.time()-start)

        ### Downsampling
        start = time.time()

        downsample = (5,5)
        
        f_R_down = block_reduce(f_R, block_size=downsample, func=np.sum)
        
        X_array_p_down = np.linspace(X_array_p.min(), X_array_p.max(), int(len(X_array_p))//downsample[0]+1)
        Y_array_p_down = np.linspace(Y_array_p.min(), Y_array_p.max(), int(len(Y_array_p))//downsample[1]+1)
        
        XX_array_p_down, YY_array_p_down = np.meshgrid(Y_array_p_down, X_array_p_down)
        print('block',time.time()-start)

        #####
        iab3_x_utm_sirgas = 203917.07880027
        iab3_y_utm_sirgas = 7545463.6805863
        
        X = XX_array_p_down + iab3_x_utm_sirgas
        Y = YY_array_p_down + iab3_y_utm_sirgas
        #####
        ### Loss
        # print(f'Loss: {(f_p.sum()- f_R_down.sum())/f_p.sum()*100:.4f} %')
        
        return (X, Y, f_R_down)

    def _array_to_raster(self, array):
        # Input array (X, Y, f)
        X = array[0]
        Y = array[1]
        f = array[2]

        # Affine transformation
        ul = (X.min(), Y.max())
        ll = (X.min(), Y.min())
        ur = (X.max(), Y.max())
        lr = (X.max(), Y.min())
        
        cols, rows = X.shape[1], X.shape[0]

        gcps = [GCP(cols, 0, *ul),
                GCP(cols, rows, *ur),
                GCP(0, 0, *ll),
                GCP(0, rows, *lr)]

        transform_out = from_gcps(gcps)

        # Output raster from memory
        with rasterio.io.MemoryFile() as memfile:
            with memfile.open(
                driver='GTiff',
                height=f.shape[1],
                width=f.shape[0],
                count=1,
                dtype=f.dtype,
                transform=transform_out,
                crs=CRS.from_epsg(31983)
            ) as dataset:
                dataset.write(f, 1)
            with memfile.open() as dataset:
                da = xarray.open_rasterio(dataset)
        return da

    def run_k15_daily(self, year, month, day):
        df_daily = self.df_k15.loc[(self.df_k15['TIMESTAMP'].dt.year==year)&
                                   (self.df_k15['TIMESTAMP'].dt.month==month)&
                                   (self.df_k15['TIMESTAMP'].dt.day==day)]
        
        print(f'{year}-{month}-{day}')
        print(f'{len(df_daily)} records')
        if len(df_daily)>0:
            metadata_datetime = df_daily['TIMESTAMP'].unique()

            if df_daily.shape[0] == 48:
                datetime_range = pd.date_range(start=f'{year}-{month}-{day}', periods=48, freq='30min')
                datetime_dataarray = xarray.DataArray(datetime_range, [('datetime', datetime_range)])
            
            else:
                datetime_range = pd.DatetimeIndex(metadata_datetime)
                datetime_dataarray = xarray.DataArray(datetime_range, [('datetime', datetime_range)])
            
            landuse = xarray.open_rasterio(self.raster_reference)

            metadata = {}
            k15_raster_list = []
            k15_individual = FFP()
            for i, row in df_daily.iterrows():
                # start = time.time()
                k15_output = k15_individual.output(zm=9,
                                                umean=row['u_rot'],
                                                h=1000,
                                                ol=row['L'],
                                                sigmav=row['v_var'],
                                                ustar=row['u*'],
                                                wind_dir=row['wind_dir_compass'],
                                                rs=None,
                                                crop=False, fig=False)
                
                start = time.time()
                warp = self._warp_array(k15_output, angle=row['wind_dir_compass'])
                print('warp_func: \t',time.time()-start)
                # start = time.time()
                k15_raster = self._array_to_raster(warp)
                # print('array_to_raster: \t',time.time()-start)
                # start = time.time()
                k15_rasterRP = k15_raster.rio.reproject_match(landuse, resampling=13)
                # print('reproject_match: \t',time.time()-start)
                k15_raster_list.append(k15_rasterRP)
            
            k15_daily = xarray.concat(k15_raster_list, datetime_dataarray)

        else:
            print(f'No data for {year}-{month}-{day}')
            k15_daily = None
            # pass
            
        return k15_daily

    def run_k15_monthly(self, year, month):
        df_monthly = self.df_k15.loc[(self.df_k15['TIMESTAMP'].dt.year==year)&
                                   (self.df_k15['TIMESTAMP'].dt.month==month)]
        
        print(f'{year}-{month}')
        print(len(df_monthly))
        if len(df_monthly)>0:
            k15_month_list = []
            for day in range(1,calendar.monthrange(year, month)[1]+1):
                # print('\t', day)
                k15_daily = self.run_k15_daily(year, month, day)
                if k15_daily is not None:
                    k15_month_list.append(k15_daily)
                else:
                    pass
            k15_month = xarray.concat(k15_month_list, dim='datetime')
            # break
        else:
            print(f'No data for {year}-{month}')
            k15_month = None
        return k15_month

    def run_k15_all(self, saveFolder):

        savePath = pathlib.Path(saveFolder)
        # print(self.df_k15['TIMESTAMP'].min().dt.year, self.df_k15['TIMESTAMP'].max())
        year_start = self.df_k15['TIMESTAMP'].min().year
        month_start = self.df_k15['TIMESTAMP'].min().month

        year_end = self.df_k15['TIMESTAMP'].max().year
        month_end = self.df_k15['TIMESTAMP'].max().month
        print(year_start, month_start)
        print(year_end, month_end)

        datetimeAll_range = pd.date_range(start=f'{year_start}-{month_start}-01', end=f'{year_end}-{month_end}-01', freq='MS')

        for datetime in datetimeAll_range:
            year = datetime.year
            month = datetime.month
            print(year, month)
            k15_month = self.run_k15_monthly(year, month)
            if k15_month is not None:
                k15_month.to_netcdf(savePath/f'k15_{year}_{month}.nc')
                print('Save to', savePath/f'k15_{year}_{month}.nc')
            else:
                print('No data for', year, month)