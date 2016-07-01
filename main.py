# http://beneathdata.com/how-to/visualizing-my-location-history/
import redis,json,csv,datetime,re
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import pandas as pd
from geopy.geocoders import Nominatim
from mpl_toolkits.basemap import Basemap
from shapely.geometry import Point, Polygon, MultiPoint, MultiPolygon,shape
from shapely.prepared import prep
from matplotlib.collections import PatchCollection
from matplotlib.colors import Normalize,LinearSegmentedColormap
from descartes import PolygonPatch
from pysal.esda.mapclassify import Natural_Breaks as nb
# must be afer Basemap...?
import fiona

def locate_place(redis_instance,geolocator,place):
		if place not in redis_instance.keys():
			location = geolocator.geocode(place)
			if location != None:
				raw_location = location.raw
				raw_location['address'] = location.address
				redis_instance.set(place,json.dumps(raw_location))
				return raw_location
			else:
				if len(place.split(',')) > 1:
					second_order_place = place[place.index(',')+1:].strip()
					if second_order_place not in redis_instance.keys():
						# remove possible event name from place string and try once more to geocode
						location = geolocator.geocode(second_order_place)
						if location != None:
							raw_location = location.raw
							raw_location['address'] = location.address
							redis_instance.set(place,json.dumps(raw_location))
							return raw_location
					else:
						redis_instance.set(place,redis_instance.get(second_order_place))
						return redis.get(second_order_place)
				else:
					print "Place could not be found: '%s'" % place
					return None

# Convenience functions for working with colour ramps and bars
def colorbar_index(ncolors, cmap, labels=None, **kwargs):
    """
    This is a convenience function to stop you making off-by-one errors
    Takes a standard colour ramp, and discretizes it,
    then draws a colour bar with correctly aligned labels
    """
    cmap = cmap_discretize(cmap, ncolors)
    mappable = cm.ScalarMappable(cmap=cmap)
    mappable.set_array([])
    mappable.set_clim(-0.5, ncolors+0.5)
    colorbar = plt.colorbar(mappable, **kwargs)
    colorbar.set_ticks(np.linspace(0, ncolors, ncolors))
    colorbar.set_ticklabels(range(ncolors))
    if labels:
        colorbar.set_ticklabels(labels)
    return colorbar

def cmap_discretize(cmap, N):
    """
    Return a discrete colormap from the continuous colormap cmap.

        cmap: colormap instance, eg. cm.jet. 
        N: number of colors.

    Example
        x = resize(arange(100), (5,100))
        djet = cmap_discretize(cm.jet, 5)
        imshow(x, cmap=djet)

    """
    if type(cmap) == str:
        cmap = get_cmap(cmap)
    colors_i = np.concatenate((np.linspace(0, 1., N), (0., 0., 0., 0.)))
    colors_rgba = cmap(colors_i)
    indices = np.linspace(0, 1., N + 1)
    cdict = {}
    for ki, key in enumerate(('red', 'green', 'blue')):
        cdict[key] = [(indices[i], colors_rgba[i - 1, ki], colors_rgba[i, ki]) for i in xrange(N + 1)]
    return LinearSegmentedColormap(cmap.name + "_%d" % N, cdict, 1024)

r = redis.StrictRedis(host='localhost', port=6379, db=0)

stats_file = 'data/field_recording_20152016.csv'

geolocator = Nominatim(timeout=3000)

data_store = dict()
data_store['lon'] = []
data_store['lat'] = []
data_store['hours'] = []
data_store['county'] = []

with open(stats_file,'rb') as csvfile:
    data = csv.reader(csvfile)
    for i,row in enumerate(data):
        if len(row[0]) and i !=0:
            date = row[0]
            event = row[1]
            location = json.loads(r.get(row[2]))
            event_type = row[3].split(',')
            hours = float(row[4])
            data_store['lon'].append(float(location['lon']))
            data_store['lat'].append(float(location['lat']))
            data_store['hours'].append(hours)
            county = re.match(ur'.*County\s(\w+).*',location['address'])
            if county != None:
                county = county.group(1).strip()
            data_store['county'].append(county )

df = pd.DataFrame(data_store)
df = df.dropna()


df.reset_index(drop=True, inplace=True)

shp = fiona.open('data/counties/test.shp')
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
    resolution='i',  suppress_ticks=True)

# m.drawcoastlines()
m.readshapefile('data/counties/test','counties', drawbounds=False,color='none', zorder=2)

plotted = []
clean_counties = []
clean_counties_info = []

for info, shape in zip(m.counties_info, m.counties):
    if info['NAME_TAG'] not in plotted:
        clean_counties.append(shape)
        clean_counties_info.append(info)
        # x, y = zip(*shape) 
        # m.plot(x, y, marker=None,color='#86A3A0')
        plotted.append(info['NAME_TAG'])

# # set up a map dataframe
df_map = pd.DataFrame({
    #access the x,y coords and define a polygon for each item in m.countes
    'poly': [Polygon(xy) for xy in clean_counties],
    #convert NAME_1 to a column called 'district'
    'county': [county['NAME_EN'] for county in clean_counties_info],
    'hours': [sum(df.loc[lambda df: (df.county == county['NAME_TAG'])]['hours']) for county in clean_counties_info]
})

# Create Point objects in map coordinates from dataframe lon and lat values
map_points = pd.Series([Point(m(mapped_x, mapped_y)) for mapped_x, mapped_y in zip(df['lon'], df['lat'])])
rec_points = MultiPoint(list(map_points.values))
counties_polygon = prep(MultiPolygon(list(df_map['poly'].values)))
county_points = filter(counties_polygon.contains, rec_points)

# Calculate Jenks natural breaks for density
breaks = nb(df_map[df_map['hours'].notnull()].hours.values,initial=300,k=32)

print df
print breaks

# the notnull method lets us match indices when joining
jb = pd.DataFrame({'jenks_bins': breaks.yb}, index=df_map[df_map['hours'].notnull()].index)
df_map = df_map.join(jb)
df_map.jenks_bins.fillna(-1, inplace=True)

# jenks_labels = ["<= %0.1f/km$^2$(%s counties)" % (b, c) for b, c in zip(breaks.bins, breaks.counts)]
# jenks_labels.insert(0, 'No field recording (%s counties)' % len(df_map[df_map['density_km'].isnull()]))

plt.clf()
fig = plt.figure()
ax = fig.add_subplot(111, axisbg='w', frame_on=False)

# use a blue colour ramp - we'll be converting it to a map using cmap()
cmap = plt.get_cmap('Blues')
# draw wards with grey outlines
df_map['patches'] = df_map['poly'].map(lambda x: PolygonPatch(x, ec='#555555', lw=.2, alpha=1., zorder=4))
pc = PatchCollection(df_map['patches'], match_original=True)
# impose our colour map onto the patch collection
norm = Normalize()
pc.set_facecolor(cmap(norm(df_map['jenks_bins'].values)))
ax.add_collection(pc)

# Add a colour bar
# cb = colorbar_index(ncolors=len(jenks_labels), cmap=cmap, shrink=0.5, labels=jenks_labels)
# cb.ax.tick_params(labelsize=6)

# # Show highest densities, in descending order
# highest = '\n'.join(value[1] for _, value in df_map[(df_map['jenks_bins'] == 10)][:10].sort().iterrows())
# highest = 'Most Dense Counties:\n\n' + highest
# # Subtraction is necessary for precise y coordinate alignment
# details = cb.ax.text(
#     -1., 0 - 0.007,
#     highest,
#     ha='right', va='bottom',
#     size=5,
#     color='#555555')

# Bin method, copyright and source data info
smallprint = ax.text(
    1.03, 0,
    'Classification method: natural breaks\nContains Ordnance Survey data\n$\copyright$ Crown copyright and database right 2013\nPlaque data from http://openplaques.org',
    ha='right', va='bottom',
    size=4,
    color='#555555',
    transform=ax.transAxes)

# Draw a map scale
m.drawmapscale(
    coords[0] + 0.025, coords[1] + 0.01,
    coords[0], coords[1],
    50.,
    barstyle='fancy', labelstyle='simple',
    fillcolor1='w', fillcolor2='#555555',
    fontcolor='#555555',
    zorder=5)
plt.title("ITMA field recording locations, Ireland")
plt.tight_layout()
# this will set the image width to 722px at 100dpi
fig.set_size_inches(7.22, 5.25)
plt.savefig('data/field_recording_locations.png', dpi=200, alpha=True)
plt.show()