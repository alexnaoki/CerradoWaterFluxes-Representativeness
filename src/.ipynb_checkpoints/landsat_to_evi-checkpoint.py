import pathlib
from turtle import end_fill
import xarray as xr
import matplotlib.pyplot as plt
import rasterio
import rioxarray
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.colors as colors

class CalculateEVI:
    def __init__(self, tif_reference, meta_reference, files_landsat, meta_files_landsat):
        print('Initialized:')

        self.raster_bool, self.raster_reference = self.reference_image(tif_reference=tif_reference, meta_reference=meta_reference)
        
        self.files_landsat = files_landsat
        self.meta_files_landsat = meta_files_landsat
        # single_landsat = files_landsat[0]
        # single_meta_landsat = meta_files_landsat[0]

        # landsat = self.calculate_evi(single_landsat, single_meta_landsat)

        # landsat_f = self.reproject_and_filter_EVI(self.raster_reference, landsat, single_meta_landsat)
        # print(landsat_f.isel(time=1)['EVI'].plot())
        
        # merged_evi = self.merge_landsat_files(files_landsat, meta_files_landsat)



    def reference_image(self, tif_reference, meta_reference='MapbiomasLULC_cerrado'):
        
        filters_img_references = {'MapbiomasLULC_cerrado': {'acc': [3, 4]}}

        if meta_reference in filters_img_references:
            # print('Reference image:', meta_reference)

            tif_file = pathlib.Path(tif_reference)
            raster_reference_pre = rasterio.open(tif_file)
            raster_reference = xr.open_rasterio(raster_reference_pre)
            raster_bool = raster_reference.isin(filters_img_references[meta_reference]['acc'])
            raster_bool = raster_bool.sel(band=1)
            # raster_bool.plot()
            # plt.show()
            print('Reference image:', meta_reference)
            return raster_bool, raster_reference

        else:
            print('Reference image not suitable')

    def calculate_evi(self, single_landsat_file, meta_landsat_file):

        config = {'Landsat7_SR': {'NIR': 'SR_B4', 'RED': 'SR_B3', 'BLUE':'SR_B1', 'scale': 0.0000275, 'offset': -0.2},
                  'Landsat8_SR': {'NIR': 'SR_B5', 'RED': 'SR_B4', 'BLUE':'SR_B2', 'scale': 0.0000275, 'offset': -0.2}}

        if meta_landsat_file in config:
            landsat = xr.open_dataset(single_landsat_file)

            landsat['NIR'] = landsat[config[meta_landsat_file]['NIR']]*config[meta_landsat_file]['scale'] + config[meta_landsat_file]['offset']
            landsat['RED'] = landsat[config[meta_landsat_file]['RED']]*config[meta_landsat_file]['scale'] + config[meta_landsat_file]['offset']
            landsat['BLUE'] = landsat[config[meta_landsat_file]['BLUE']]*config[meta_landsat_file]['scale'] + config[meta_landsat_file]['offset']

            landsat['EVI'] = 2.5*(landsat['NIR']-landsat['RED'])/(landsat['NIR']+6*landsat['RED']-7.5*landsat['BLUE']+1)

            landsat['EVI'] = landsat['EVI'].rio.write_crs('epsg:31983')

            return landsat
        else:
            print('Landsat file not suitable')

    def reproject_and_filter_EVI(self, raster_bool, landsat, meta_landsat_file):

        config = {'Landsat7_SR': {'QA_PIXEL': [5440], 'QA_RADSAT': [0]},
                  'Landsat8_SR': {'QA_PIXEL': [21824], 'QA_RADSAT': [0]}}

        # Reproject to same resolution as reference image
        landsat_r = landsat.rio.reproject_match(raster_bool)

        # Filter out pixels with bad quality
        if meta_landsat_file in config:
            landsat_f = landsat_r.where((landsat_r['QA_PIXEL'].isin(config[meta_landsat_file]['QA_PIXEL']) & landsat_r['QA_RADSAT'].isin(config[meta_landsat_file]['QA_RADSAT'])))
            landsat_f = landsat_f.where((landsat_f['EVI']<=1) & (landsat_f['EVI']>=-1))

            return landsat_f

        else:
            print('Landsat file not suitable')

    def merge_landsat_files(self):
        evis = []
        
        for file, meta in zip(self.files_landsat, self.meta_files_landsat):
            print(file, meta)

            landsat_evi = self.calculate_evi(file, meta)
            landsat_evi_f = self.reproject_and_filter_EVI(self.raster_reference, landsat_evi, meta)
            evis.append(landsat_evi_f)

        self.evi = xr.merge(evis)
        print('Files merged')
        return self.evi


    def interpolate_to_daily(self, target_year):

        start_pad = f'{target_year-1}-12-01'
        end_pad = f'{target_year+1}-03-01'

        start = f'{target_year}-01-01'
        end = f'{target_year}-12-31'

        self.start = start
        self.end = end

        evi_s = self.evi.sel(time=slice(start_pad, end_pad))
        # evi_s['EVI'].mean(dim=('x','y')).plot()
        # plt.show()

    
        evi_s['EVI_interpolated'] = evi_s['EVI'].interpolate_na(dim='time')
        # evi_s['EVI_interpolated'].sel(time='2020-12-24').plot()
        # plt.show()

        evi_i = evi_s['EVI_interpolated'].interp(time=pd.date_range(start=start_pad, end=end_pad, freq='1d'))
        
        self.evi_is = evi_i.sel(time=slice(start, end))
        # evi_is.sel(time='2020-12-30').plot()
        self.evi_is.mean(dim=('x','y')).plot()
        plt.show()
        return self.evi_is

    def interpolate_to_30min(self):
        
        
        # evi_s = self.evi_is.interpolate_na(dim='time')

        # print(evi_s)
        evi_s = self.evi_is.sel(time=slice('2020-01-01','2020-01-07'))
        evi_i = evi_s.interp(time=pd.date_range(start='2020-01-01', end='2020-01-07', freq='30min'))
        print(evi_i)
        return evi_i