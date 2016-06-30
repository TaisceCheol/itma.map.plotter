# http://beneathdata.com/how-to/visualizing-my-location-history/

import redis,json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mpl_toolkits.basemap import Basemap
from shapely.geometry import Point, Polygon, MultiPoint, MultiPolygon,shape
from shapely.prepared import prep
import fiona
from matplotlib.collections import PatchCollection
from descartes import PolygonPatch
import json
import datetime
from random import randint

def num_of_contained_points(apolygon, points):
    return int(len(filter(prep(apolygon).contains, points)))

r = redis.StrictRedis(host='localhost', port=6379, db=0)
placelist = ['Dublin','Westport','Miltown Malbay','Glencolumcille, Donegal','Leitrim','Blarney, Cork','Rathlin Island','Ballybofey, Donegal','Navan',"John Doherty's grave, Lough Finn, Donegal"]
places = [json.loads(r.get(x)) for x in placelist]

output = dict()
output['lon'] = []
output['lat'] = []
output['hours'] = []

for p in places:
	output['lon'].append(p['lon'])
	output['lat'].append(p['lat'])
	output['hours'].append(randint(1,24))

df = pd.DataFrame(output)
df = df.dropna()

df[['lon', 'lat','hours']] = df[['lon', 'lat','hours']].astype(float)

df.reset_index(drop=True, inplace=True)

shp = fiona.open('data/counties/counties.shp')
coords = shp.bounds
shp.close()
w, h = coords[2] - coords[0], coords[3] - coords[1]
extra = 0.1

m = Basemap(
    projection='tmerc', ellps='WGS84',
    lon_0=-8,
    lat_0=53.5,
    llcrnrlon=coords[0] - extra * w,
    llcrnrlat=coords[1] - (extra * h), 
    urcrnrlon=coords[2] + extra * w,
    urcrnrlat=coords[3] + (extra * h),
    resolution='i',  suppress_ticks=True,epsg=29902)

m.drawcoastlines()
m.readshapefile('data/counties/counties','counties',color='none',zorder=2)

# # set up a map dataframe
df_map = pd.DataFrame({
    #access the x,y coords and define a polygon for each item in m.countes
    'poly': [Polygon(xy) for xy in m.counties],
    #convert NAME_1 to a column called 'district'
    'county': [county['NAME_EN'] for county in m.counties_info]})

print df_map
# Create Point objects in map coordinates from dataframe lon and lat values
map_points = [Point(m(mapped_x, mapped_y)) for mapped_x, mapped_y in zip(df['lon'], df['lat'])]

recording_points = MultiPoint(map_points)

counties_polygon = prep(MultiPolygon(list(df_map['poly'].values)))

county_points = filter(counties_polygon.contains, recording_points)

df_map['county_count'] = df_map['poly'].apply(num_of_contained_points,args=(county_points,))


