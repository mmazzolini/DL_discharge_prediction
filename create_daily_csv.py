import pandas as pd
import numpy as np
import xarray as xr
import geopandas as gpd
from shapely.geometry import Point, Polygon
import matplotlib.pyplot as plt
import psycopg2
import shapely.wkb as wkb
from geopandas import GeoSeries

import pdb

def check_data_gap(df):

    df.dropna(axis='rows', how='all', inplace=True)
    missing_dates = pd.date_range(df.index.min(), df.index.max()).difference(df.index)
    
    print(f'Date start: {df.index.min().strftime("%Y-%m-%d")}, date end: {df.index.max().strftime("%Y-%m-%d")}')

    if len(missing_dates) > 0:
        print(f"Missing dates: {', '.join(missing_dates.strftime('%Y-%m-%d'))}")
    else:
        print('No missing dates')
    # return missing_dates


def interpolate_df(df):

    df = df.reindex(pd.date_range(df.index.min(), df.index.max()), fill_value=np.nan)
    return df.interpolate()


def readnetcdf_in_shp_mattia(nc_fileName, shp_fileName, res=0.25, plot=False):

    # Opent the netcdf file
    ds = xr.open_dataset(nc_fileName)

    # Open the shape file and reproject it to lat lon WGS84
    shp = gpd.read_file(shp_fileName)
    shp = shp.to_crs('epsg:4326')

    # Crop ds with the shapefile bounding box (bb)
    bb = shp.bounds.iloc[0]
    ds = ds.sel(lon=slice(bb['minx']-res, bb['maxx']+res), lat=slice(bb['maxy']+res, bb['miny']-res))

    # Mask all the points in ds where the grid box do not intersect or is in the shapefile
    for x in ds.longitude.values:
        for y in ds.latitude.values:
            gridbox = Point(x, y).buffer(res/2, cap_style=3)
            if not gridbox.intersects(shp.loc[0, 'geometry']):
                for k in ds.data_vars.keys():
                    ds[k].loc[dict(longitude=x, latitude=y)] = np.nan
    ds = ds.dropna(dim='longitude', how='all')
    ds = ds.dropna(dim='latitude', how='all')

    # Plot the era5 gridbox and the shapefile if plot=True
    if plot:
        for x in ds.longitude.values:
            for y in ds.latitude.values:
                gridbox = Point(x, y).buffer(res / 2, cap_style=3)
                gridbox_x, gridbox_y = gridbox.exterior.xy
                plt.plot(gridbox_x, gridbox_y, color='blue')
                plt.plot(x, y, marker='o', color='red')
        shp_x, shp_y = shp.loc[0, 'geometry'].exterior.xy
        plt.plot(shp_x, shp_y, color='black')
        plt.axis('equal')

    return ds


def readnetcdf_in_shp(nc_fileName, shp_fileName, res=5500, plot=False):
    
    # Open the netcdf file
    ds = xr.open_dataset(nc_fileName)
    
    # Open the shape file and reproject it to the MESCAN-Surfex grid (unit=meters)
    shp = gpd.read_file(shp_fileName)
    shp_reproj = shp.to_crs('+proj=lcc +lat_1=50 +lat_2=50 +lat_0=50 +lon_0=8 +x_0=2937018.5829291 +y_0=2937031.41074803 +a=6371229 +b=6371229')
    
    # Crop ds with the shapefile bounding box (bb)
    bb = shp_reproj.bounds.iloc[0]
    ds = ds.sel(x=slice(bb['minx']-res, bb['maxx']+res), 
                y=slice(bb['miny']-res, bb['maxy']+res))
    
    
    #0000 Mask all the points in ds where the grid box do not intersect or is in the shapefile
    for i in ds.x.values:
        for j in ds.y.values:
            gridbox = Point(i, j).buffer(res/2, cap_style=3)
            if not (gridbox.intersects(shp_reproj.loc[0, 'geometry'])):
                for k in ds.data_vars.keys():
                    if not (k =='Lambert_Conformal'):
                        ds[k].loc[dict(x=i, y=j)] = np.nan
    ds = ds.dropna(dim='x', how='all')
    ds = ds.dropna(dim='y', how='all')

    counter=0                    
    # Plot the era5 gridbox and the shapefile if plot=True
    if plot:
        for x in ds.x.values:
            for y in ds.y.values:
                gridbox = Point(x, y).buffer(res / 2, cap_style=3)
                gridbox_x, gridbox_y = gridbox.exterior.xy
                plt.plot(gridbox_x, gridbox_y, color='blue')
                for k in ds.data_vars.keys():
                    if (k !='Lambert_Conformal'):
                        if not(ds[k].loc[dict(x=x, y=y)].isnull().all()):
                            plt.plot(x, y, marker='o', color='red')
                            counter=counter+1
        shp_x, shp_y = shp_reproj.loc[0, 'geometry'].exterior.xy
        plt.plot(shp_x, shp_y, color='black')
        plt.axis('equal')                        
    print(f'n of pixels{counter}')                  
    return ds


def readnetcdf_in_shp_db(nc_fileName, STAT_CODE, res=5500, plot=False):
    
    # Open the netcdf file
    ds = xr.open_dataset(nc_fileName)
    
    # Open the shape file and reproject it to the MESCAN-Surfex grid (unit=meters)

    conn = psycopg2.connect(host="10.8.244.31",
                       database="climate_data",
                       user="ado_user",
                       password="hydro#ado",
                       port=5432)
    cur = conn.cursor()
    
    # get the metadata
    query = f"""
            SELECT "geom" FROM "hydrology"."catchment_area" WHERE "id_station" = '{STAT_CODE}'    
            """
    df = pd.read_sql_query(query, conn)
    
    # close the connection when finished
    cur.close()
    conn.close()
    
    
    shp=GeoSeries(wkb.loads(df.geom[0], hex=True))
    shp=shp.set_crs("EPSG:4326")
    shp_reproj = shp.to_crs('+proj=lcc +lat_1=50 +lat_2=50 +lat_0=50 +lon_0=8 +x_0=2937018.5829291 +y_0=2937031.41074803 +a=6371229 +b=6371229')
    
    # Crop ds with the shapefile bounding box (bb)
    bb = shp_reproj.bounds.iloc[0]
    ds = ds.sel(x=slice(bb['minx']-res, bb['maxx']+res), 
                y=slice(bb['miny']-res, bb['maxy']+res))
    
    
    #0000 Mask all the points in ds where the grid box do not intersect or is in the shapefile
    for i in ds.x.values:
        for j in ds.y.values:
            gridbox = Point(i, j)#.buffer(res/2, cap_style=3)
            if not (gridbox.intersects(shp_reproj.loc[0])):
                for k in ds.data_vars.keys():
                    if not (k =='Lambert_Conformal'):
                        ds[k].loc[dict(x=i, y=j)] = np.nan
    ds = ds.dropna(dim='x', how='all')
    ds = ds.dropna(dim='y', how='all')
    counter=0                    
    # Plot the era5 gridbox and the shapefile if plot=True
    if plot:
        plt.figure(figsize=(25,25))
        for x in ds.x.values:
            for y in ds.y.values:
                gridbox = Point(x, y).buffer(res / 2, cap_style=3)
                gridbox_x, gridbox_y = gridbox.exterior.xy
                plt.plot(gridbox_x, gridbox_y, color='blue')
                for k in ds.data_vars.keys():
                    if (k !='Lambert_Conformal'):
                        if not(ds[k].loc[dict(x=x, y=y)].isnull().all()):
                            plt.plot(x, y, marker='o', color='red')
                            counter=counter+1
        
        coords=[p.exterior.xy for p in shp_reproj.loc[0]]
        shp_x=coords[0][0]
        shp_y=coords[0][1]
        #shp_x, shp_y = *shp_reproj.loc[0].exterior.xy
        plt.plot(shp_x, shp_y, color='black')
        plt.axis('equal')                        
    print(f'n of pixels{counter}')  
    
    return ds


def xarray2df(xa, varnamedest,varnameor=False):
    if not varnameor:
        df = {}
        for i in range(xa.y.size):
            for j in range(xa.x.size):
                df[f'{varnamedest}x{j}y{i}'] = xa.isel(y=i, x=j).to_dataframe().iloc[:, 2]
                #pdb.set_trace()

    else:
        df = {}
        for i in range(xa.y.size):
            for j in range(xa.x.size):
                df[f'{varnamedest}x{j}y{i}'] = xa.isel(y=i, x=j).to_dataframe().loc[:,varnameor]
                #pdb.set_trace()

    frame=pd.DataFrame(df)
    return frame


def get_discharge_from_DB(STAT_CODE):
    #read the csv file with the daily discharge

    # establish connection using information supplied in documentation
    conn = psycopg2.connect(host="10.8.244.31",
                           database="climate_data",
                           user="ado_user",
                           password="hydro#ado",
                           port=5432)
    cur = conn.cursor()
    
    # get the metadata
    query = f"""
    SELECT date, discharge_m3_s FROM hydrology.discharge WHERE id_station = '{STAT_CODE}' ORDER BY date;
    """

    df = pd.read_sql_query(query, conn)
   
    #set the date in a proper format
    df.index = pd.to_datetime(df.date)
    df.drop(columns='date',inplace=True)
    # close the connection when finished
    cur.close()
    conn.close()
    return df

# ------------------------------------------------------------------------------------------------------------------

