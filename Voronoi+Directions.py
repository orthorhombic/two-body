from datetime import datetime
from matplotlib.path import Path
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import cartopy
import cartopy.crs as ccrs
from cartopy.io.img_tiles import OSM
import pandas
import matplotlib
import seaborn
import googlemaps
from datetime import datetime
from scipy.spatial import Voronoi, voronoi_plot_2d
import shapely.geometry
import shapely.ops
import time



gmaps = googlemaps.Client(key='AIzaSyCEVlqLL37Sy3D7ASUTjiy6WxNCTtUOVNA')


#Decode polyline from route to plot
#https://gist.github.com/signed0/2031157

'''Provides utility functions for encoding and decoding linestrings using the 
Google encoded polyline algorithm.
'''

def encode_coords(coords):
    '''Encodes a polyline using Google's polyline algorithm
    
    See http://code.google.com/apis/maps/documentation/polylinealgorithm.html 
    for more information.
    
    :param coords: Coordinates to transform (list of tuples in order: latitude, 
    longitude).
    :type coords: list
    :returns: Google-encoded polyline string.
    :rtype: string    
    '''
    
    result = []
    
    prev_lat = 0
    prev_lng = 0
    
    for x, y in coords:        
        lat, lng = int(y * 1e5), int(x * 1e5)
        
        d_lat = _encode_value(lat - prev_lat)
        d_lng = _encode_value(lng - prev_lng)        
        
        prev_lat, prev_lng = lat, lng
        
        result.append(d_lat)
        result.append(d_lng)
    
    return ''.join(c for r in result for c in r)
    
def _split_into_chunks(value):
    while value >= 32: #2^5, while there are at least 5 bits
        
        # first & with 2^5-1, zeros out all the bits other than the first five
        # then OR with 0x20 if another bit chunk follows
        yield (value & 31) | 0x20 
        value >>= 5
    yield value

def _encode_value(value):
    # Step 2 & 4
    value = ~(value << 1) if value < 0 else (value << 1)
    
    # Step 5 - 8
    chunks = _split_into_chunks(value)
    
    # Step 9-10
    return (chr(chunk + 63) for chunk in chunks)

def decode(point_str):
    '''Decodes a polyline that has been encoded using Google's algorithm
    http://code.google.com/apis/maps/documentation/polylinealgorithm.html
    
    This is a generic method that returns a list of (latitude, longitude) 
    tuples.
    
    :param point_str: Encoded polyline string.
    :type point_str: string
    :returns: List of 2-tuples where each tuple is (latitude, longitude)
    :rtype: list
    
    '''
            
    # sone coordinate offset is represented by 4 to 5 binary chunks
    coord_chunks = [[]]
    for char in point_str:
        
        # convert each character to decimal from ascii
        value = ord(char) - 63
        
        # values that have a chunk following have an extra 1 on the left
        split_after = not (value & 0x20)         
        value &= 0x1F
        
        coord_chunks[-1].append(value)
        
        if split_after:
                coord_chunks.append([])
        
    del coord_chunks[-1]
    
    coords = []
    
    for coord_chunk in coord_chunks:
        coord = 0
        
        for i, chunk in enumerate(coord_chunk):                    
            coord |= chunk << (i * 5) 
        
        #there is a 1 on the right if the coord is negative
        if coord & 0x1:
            coord = ~coord #invert
        coord >>= 1
        coord /= 100000.0
                    
        coords.append(coord)
    
    # convert the 1 dimensional list to a 2 dimensional list and offsets to 
    # actual values
    points = []
    prev_x = 0
    prev_y = 0
    for i in range(0, len(coords) - 1, 2):
        if coords[i] == 0 and coords[i + 1] == 0:
            continue
        
        prev_x += coords[i + 1]
        prev_y += coords[i]
        # a round to 6 digits ensures that the floats are the same as when 
        # they were encoded
        points.append((round(prev_x, 6), round(prev_y, 6)))
    
    return points    


def voronoi_finite_polygons_2d(vor, radius=None):
    """
    Reconstruct infinite voronoi regions in a 2D diagram to finite
    regions.

    Parameters
    ----------
    vor : Voronoi
        Input diagram
    radius : float, optional
        Distance to 'points at infinity'.

    Returns
    -------
    regions : list of tuples
        Indices of vertices in each revised Voronoi regions.
    vertices : list of tuples
        Coordinates for revised Voronoi vertices. Same as coordinates
        of input vertices, with 'points at infinity' appended to the
        end.

    """

    if vor.points.shape[1] != 2:
        raise ValueError("Requires 2D input")

    new_regions = [0]*(len(vor.point_region)+1) #create new regions var in same format as original to correlate points to regions
    new_vertices = vor.vertices.tolist()

    center = vor.points.mean(axis=0)
    if radius is None:
        radius = vor.points.ptp(axis=0).max()

    # Construct a map containing all ridges for a given point
    #dictionary references each input point by index (from ridge_points) and compiles a list of relevant ridges
    #format: key=input point index, (point2, vertex1, vertex2)
    all_ridges = {} 
    for (p1, p2), (v1, v2) in zip(vor.ridge_points, vor.ridge_vertices): #each ridge expressed as points
        #print ((p1, p2), (v1, v2))
        all_ridges.setdefault(p1, []).append((p2, v1, v2))
        all_ridges.setdefault(p2, []).append((p1, v1, v2))

    # Reconstruct infinite regions
    for p1, region in enumerate(vor.point_region):
#        print (p1, region) #output the data point index and region index
        vertices = vor.regions[region] #index of voronoi vertices
        
        #if distance from centroid is greater than 2*radius, set point to 'infinity'
        for i in range(len(vertices)):
#            print ('i= ',i)
#            print('Vertex # ',vertices[i]) #Note: need to set vertex 0 as infinity
            if vertices[i] >= 0:
                #print vert coord
#                print('vert coord ',new_vertices[vertices[i]])
                #print point coord
#                print ('point coord ',vor.points[p1])
    
                #print distance
                fD=np.sqrt( (vor.points[p1][0] - new_vertices[vertices[i]][0])**2 + (vor.points[p1][1] - new_vertices[vertices[i]][1])**2 )
#                print('Distance: ',fD)
                
                #if distance >2*radius, set vertices[i]=-1
                if fD > 2*radius:
#                    print('Vertex too far: ',vertices[i])
                     
                    #remove vertex from all_ridges[p1] by setting -1
                    for j in range(len(all_ridges[p1])):
#                        print (j)
#                        print ('Vertex indices: ',all_ridges[p1][j][1], all_ridges[p1][j][2])
    
                        if all_ridges[p1][j][1] == vertices[i]:
#                            print('v1: ', all_ridges[p1][j][1])
#                            print('old: ',(all_ridges[p1][j][0], all_ridges[p1][j][1], all_ridges[p1][j][2]))
#                            print('new: ',(all_ridges[p1][j][0], -1, all_ridges[p1][j][2]))
                            all_ridges[p1][j]=(all_ridges[p1][j][0], -1, all_ridges[p1][j][2])
#                            print('new2: ',all_ridges[p1])
                        if all_ridges[p1][j][2] == vertices[i]:
#                            print('v2: ', all_ridges[p1][j][2])
#                            print('old: ',(all_ridges[p1][j][0], all_ridges[p1][j][1], all_ridges[p1][j][2]))
#                            print('new: ',(all_ridges[p1][j][0], all_ridges[p1][j][1], -1))
                            all_ridges[p1][j]=(all_ridges[p1][j][0], all_ridges[p1][j][1], -1)
    
    
    
#                        print('')
                    #set vertex to -1
                    vertices[i]=-1
                        
                
                
#            print ('')

        if all(v >= 0 for v in vertices): #check for real vertices
            # finite region
#            print('finite')
            #new_regions.append(vertices)
            new_regions[region]=(vertices)
            continue

        # reconstruct a non-finite region
        ridges = all_ridges[p1]
        new_region = [v for v in vertices if v >= 0]

#which v1/v2 ridge vertex corespond to voronoi vertex? same zero?


        for p2, v1, v2 in ridges:
#            print (p2, v1, v2)
            if v2 < 0:
                v1, v2 = v2, v1 #switch to make v1 infintie, v2 finite
            if v1 >= 0:
                # finite ridge: already in the region
                continue
            if (v1 < 0 and v2 <0):
                # not real ridge: skip
                #print ('continue',  (p2, v1, v2))
                continue
            # Compute the missing endpoint of an infinite ridge

            t = vor.points[p2] - vor.points[p1] # tangent vector
            t /= np.linalg.norm(t) # normalize vector (unit vector)
            n = np.array([-t[1], t[0]])  # normal, unit vector

            midpoint = vor.points[[p1, p2]].mean(axis=0)
            # determine sign for direction, apply to unit vector
            direction = np.sign(np.dot(midpoint - center, n)) * n
            #create vector following normal from the voronoi vertex with length of radius
            far_point = vor.vertices[v2] + direction * radius

            new_region.append(len(new_vertices))
            new_vertices.append(far_point.tolist())

        # sort region counterclockwise
        vs = np.asarray([new_vertices[v] for v in new_region])
        c = vs.mean(axis=0)
        angles = np.arctan2(vs[:,1] - c[1], vs[:,0] - c[0])
        new_region = np.array(new_region)[np.argsort(angles)]

        # finish
        new_regions[region]=(new_region.tolist())

    return new_regions, np.asarray(new_vertices)


#dataframe of destination information
asColumns=[
        'Lat',
        'Lon',
        'Arrival',
        'Depart',
        'NumDays'
        ]

dfDest=pandas.DataFrame(columns=asColumns)

#West
dfDest.loc[0,'Lat']=40.061341
dfDest.loc[0,'Lon']=-75.680189
dfDest.loc[0,'Arrival']=pandas.datetime(2018,2,6,8,15) # do not use leading zeros
dfDest.loc[0,'Depart']=pandas.datetime(2018,2,6,17,10)
dfDest.loc[0,'NumDays']=5
#Towson
dfDest.loc[1,'Lat']=39.392512
dfDest.loc[1,'Lon']=-76.612639
dfDest.loc[1,'Arrival']=pandas.datetime(2018,2,6,11,15)
dfDest.loc[1,'Depart']=pandas.datetime(2018,2,6,15,30)
dfDest.loc[1,'NumDays']=3.56
##Towson Northeast
#dfDest.loc[1,'Lat']=39.564511
#dfDest.loc[1,'Lon']=-76.289427
#dfDest.loc[1,'Arrival']=pandas.datetime(2018,2,6,11,15)
#dfDest.loc[1,'Depart']=pandas.datetime(2018,2,6,15,30)
#dfDest.loc[1,'NumDays']=3.5



##Define square (not rotated) 
#would actually be best to define circle which encompasses all points
origin=np.array([0,0])
#corner=afDest2-afDest1
corner=[
    dfDest.loc[:,'Lon'].max()-dfDest.loc[:,'Lon'].min(),
    dfDest.loc[:,'Lat'].max()-dfDest.loc[:,'Lat'].min()
    ]

#pythagorean theorm for distance, 45-45-90 triangle for ratio to side
hypotenuse=np.sqrt((corner[0]**2+corner[1]**2)) 
side=hypotenuse/np.sqrt(2) 
#specify spacing for side
#note: should implement dynamic scaling: between minimum useful point distance and maximum number of api calls
spacing=np.linspace(0,side,num=15,endpoint=True)

#Create an initial grid 
 #Create a mesh grid
 #Transpose to make each lat/lon point into a row
 #reshape to combine multiple arrays into one
aInitGrid=np.array(np.meshgrid(spacing, spacing, indexing='xy')).T.reshape(-1, 2)


# dataframe of (lon/lat) pairs
#skip two end points
dfGrid=pandas.DataFrame(aInitGrid[1:-1],columns=['RelLon','RelLat']) 


##do the math to rotate the grid and reset origin
#calculate rotation of square relative to lat/lon, subtract 45deg inside square
fTheta=np.arcsin(corner[1]/hypotenuse)-(np.pi/4) #use to avoid a divide by zero possibility

#rotate grid
dfGrid['Lon']=dfGrid['RelLon']*np.cos(fTheta)-dfGrid['RelLat']*np.sin(fTheta)
dfGrid['Lat']=dfGrid['RelLon']*np.sin(fTheta)+dfGrid['RelLat']*np.cos(fTheta)

#move back to origin
dfGrid['CorrLon']=dfGrid['Lon']+dfDest.loc[:,'Lon'].min()
dfGrid['CorrLat']=dfGrid['Lat']+dfDest.loc[:,'Lat'].min()
##############


#shortest route between two jobs
#departure time is arbitrarily the one selected for the first job


gmDir_jobjob = gmaps.directions(
        origin=(dfDest.loc[0,'Lat'],dfDest.loc[0,'Lon']),
        destination=(dfDest.loc[1,'Lat'],dfDest.loc[1,'Lon']),
         mode='driving',
#         #        alternatives=False
##        avoid=
#         units='imperial',
         departure_time=dfDest.loc[0,'Depart'],
         traffic_model="best_guess"
         )


#gmDir_jobjob[0]['bounds'] #this might be a good way of defining the window size for initial plot
#min lon 
fMinLon=np.array([dfGrid['CorrLon'].min(), gmDir_jobjob[0]['bounds']['southwest']['lng'] ]).min()
#max lon
fMaxLon=np.array([dfGrid['CorrLon'].max(), gmDir_jobjob[0]['bounds']['northeast']['lng'] ]).max()
#difference
fDeltalon=fMaxLon-fMinLon


#min lat
#min lon buffer 2.5% of difference
fMinLat=np.array([dfGrid['CorrLat'].min(), gmDir_jobjob[0]['bounds']['southwest']['lat'] ]).min()
#max lon
fMaxLat=np.array([dfGrid['CorrLat'].max(), gmDir_jobjob[0]['bounds']['northeast']['lat'] ]).max()
#difference
fDeltalat=fMaxLat-fMinLat

fBuffer=0.04 #buffer % of difference

aExtent=[
        fMinLon-fDeltalon*fBuffer,
        fMaxLon+fDeltalon*fBuffer,
        fMinLat-fDeltalat*fBuffer,
        fMaxLat+fDeltalat*fBuffer,
        ]

#decode direction route
aRoute=np.array(decode(gmDir_jobjob[0]['overview_polyline']['points']))

#display/printcopyright and warning info
print(gmDir_jobjob[0]['copyrights']) # [0] to return first direction option
print('Warnings: ',gmDir_jobjob[0]['warnings'])

###############plotting
#imagery = OSM() # Use Open street maps data
#ax = plt.axes(projection=imagery.crs)
##ax.set_extent([-78, -74, 38, 41], ccrs.Geodetic()) #longitude, latitude (x1,x2,y1,y2)
#ax.set_extent(aExtent, ccrs.Geodetic()) #longitude, latitude (x1,x2,y1,y2)
#
## Add the imagery to the map. Later iterations will need to intellegently determine zoom level
#ax.add_image(imagery, 7) #good
#
##Destinations
#plt.plot(dfDest.loc[:,'Lon'], dfDest.loc[:,'Lat'],
#         marker='o', color='blue', markersize=9, transform=ccrs.Geodetic(),
#         linestyle='')
#
##plot coarse grid
#plt.plot(dfGrid['CorrLon'],dfGrid['CorrLat'],
#         marker='o', color='black', markersize=1, transform=ccrs.Geodetic(),
#         linestyle='')
#
#plt.plot(
#        aRoute[:,0], #x lng
#        aRoute[:,1], #y lat
#         marker='o', linestyle='--', color='green', markersize=1, transform=ccrs.Geodetic()
#         )


#convert dataframe grid to multiindex
#ideally move earlier in code

indNew=pandas.MultiIndex(
        
        levels= [
                [0],
                dfGrid.columns
                ], #outside/first, inside/second (names)
        labels= [
                [0]*len(dfGrid.columns),
                [0,1,2,3,4,5]
                ],#array of level indicies corresponding to data
        names= ['Base','DestInt'], #name levels, first is outside
        )
dfGrid.columns=indNew



#for each point:
for i in range(len(dfGrid)): #whole df
    print ('row i: ',i) #row of initial point
    #for each destination:

    for j in range(len(dfDest)): #whole df
        print ('j: ',j) #row of destination/column of grid
        
        #Compute depart directions
        gmDir_Dtemp = gmaps.directions(
                origin=(dfDest.loc[j,'Lat'],dfDest.loc[j,'Lon']), #dest   lat,lon ,or string array with address
                destination=(dfGrid.loc[i,(0,'CorrLat')],dfGrid.loc[i,(0,'CorrLon')]), #grid pt   lat,lon ,or string array with address
                 mode='driving',
                 departure_time=dfDest.loc[j,'Depart'],
                 traffic_model="best_guess"
                 )    

        #while loop to catch errors from google maps - perhaps this should be converted into a function
        while (
               'overview_polyline' not in gmDir_Dtemp[0] or
               'duration' not in gmDir_Dtemp[0]['legs'][0] or #dur in seconds
               'duration_in_traffic' not in gmDir_Dtemp[0]['legs'][0] #dur in seconds
               ):
            print('Retrying in 5 seconds')        
            time.sleep(5)
            
            gmDir_Dtemp = gmaps.directions(
                    origin=(dfDest.loc[j,'Lat'],dfDest.loc[j,'Lon']), #dest   lat,lon ,or string array with address
                    destination=(dfGrid.loc[i,(0,'CorrLat')],dfGrid.loc[i,(0,'CorrLon')]), #grid pt   lat,lon ,or string array with address
                     mode='driving',
                     departure_time=dfDest.loc[j,'Depart'],
                     traffic_model="best_guess"
                     )    
            
            
            
    
        ##store departure columns from leg 0
        #store general depart data
        dfGrid.at[i,(j,'D-Polyline')]  =(
                 gmDir_Dtemp[0]['overview_polyline']['points']
                 )#keep as polyline for time
         
        dfGrid.at[i,(j,'D-dur')]=(
                 gmDir_Dtemp[0]['legs'][0]['duration']['value'] #dur in seconds
                 )

        dfGrid.at[i,(j,'D-dur_traffic')]=(
                 gmDir_Dtemp[0]['legs'][0]['duration_in_traffic']['value'] #dur in seconds
                 )
         
        dfGrid.at[i,(j,'D-traffic')]=(
                 dfGrid.at[i,(j,'D-dur_traffic')]-dfGrid.at[i,(j,'D-dur')]
                 )

        #Compute Arrival directions
        #########Need to readjust to calculate departure time
        #########Maps API will not calculate traffic without a departure time set
        #specified arrival - return trip time(seconds)        
        tdArrivalShift=pandas.Timedelta(
                dfGrid.at[i,(j,'D-dur_traffic')] 
                ,unit='s')
        
        #use departure transit time to estimate necessary departure time
        gmDir_Atemp = gmaps.directions(
                origin=(dfGrid.loc[i,(0,'CorrLat')],dfGrid.loc[i,(0,'CorrLon')]), #grid pt   lat,lon ,or string array with address
                destination=(dfDest.loc[j,'Lat'],dfDest.loc[j,'Lon']), #dest   lat,lon ,or string array with address
                 mode='driving',
        #         #        alternatives=False
        ##        avoid=
        #         units='imperial',
                 departure_time=dfDest.loc[j,'Arrival']-tdArrivalShift,
#                 traffic_model="best_guess"
                 )    
        
        
        #while loop to catch errors from google maps - perhaps this should be converted into a function
        while (
               'overview_polyline' not in gmDir_Atemp[0] or
               'duration' not in gmDir_Atemp[0]['legs'][0] or #dur in seconds
               'duration_in_traffic' not in gmDir_Atemp[0]['legs'][0] #dur in seconds
               ):
            print('Retrying in 5 seconds')        
            time.sleep(5)
            gmDir_Atemp = gmaps.directions(
                    origin=(dfGrid.loc[i,(0,'CorrLat')],dfGrid.loc[i,(0,'CorrLon')]), #grid pt   lat,lon ,or string array with address
                    destination=(dfDest.loc[j,'Lat'],dfDest.loc[j,'Lon']), #dest   lat,lon ,or string array with address
                     mode='driving',
            #         #        alternatives=False
            ##        avoid=
            #         units='imperial',
                     departure_time=dfDest.loc[j,'Arrival']-tdArrivalShift,
    #                 traffic_model="best_guess"
                     )                
            

        ##store arrival columns from leg 0
        if j==0:
            #store start location
             dfGrid.at[i,(j,'GMLon')]   =(
                     gmDir_Atemp[0]['legs'][0]['start_location']['lng']
                     )
             dfGrid.at[i,(j,'GMLat')]=(
                     gmDir_Atemp[0]['legs'][0]['start_location']['lat']
                     )        

        #store general arrival data
        dfGrid.at[i,(j,'A-Polyline')]  =(
                 gmDir_Atemp[0]['overview_polyline']['points']
                 )#keep as polyline for time
         
        dfGrid.at[i,(j,'A-dur')]=(
                 gmDir_Atemp[0]['legs'][0]['duration']['value'] #dur in seconds
                 )

        dfGrid.at[i,(j,'A-dur_traffic')]=(
                 gmDir_Atemp[0]['legs'][0]['duration_in_traffic']['value'] #dur in seconds
                 )
         
        dfGrid.at[i,(j,'A-traffic')]=(
                 dfGrid.at[i,(j,'A-dur_traffic')]-dfGrid.at[i,(j,'A-dur')]
                 )

        #########Daily Destination Subtotal        
        #total commute time, multiploed by days/week
        dfGrid.at[i,(j,'SubTot_dur_traffic')]=(
                 dfGrid.at[i,(j,'A-dur_traffic')]+
                 dfGrid.at[i,(j,'D-dur_traffic')]#dur in seconds
                 ) * dfDest.loc[j,'NumDays']
        #total extra spent in traffic
        dfGrid.at[i,(j,'SubTot_traffic')]=(
                 dfGrid.at[i,(j,'A-traffic')]+
                 dfGrid.at[i,(j,'D-traffic')]#dur in seconds
                 ) * dfDest.loc[j,'NumDays']
    
#Final point evaluation - all driving for all destinations
#TotCom - total commute time for all destinations
dfGrid.loc[:,(0,'TotCom')]=(
    dfGrid.loc[
            :, #dataframe row
            pandas.IndexSlice[:, 'SubTot_dur_traffic']
            ].sum(axis=1)
    )
#TotTraf #all extra traffic for all destinations
dfGrid.loc[:,(0,'TotTraf')]=(
        dfGrid.loc[
        :, #dataframe row
        pandas.IndexSlice[:, 'SubTot_traffic']
        ].sum(axis=1)
    )
#standard deviation to measure equity/fairness of driving
dfGrid.loc[:,(0,'Equity')]=(
        dfGrid.loc[
            :, #dataframe row
            pandas.IndexSlice[:, 'SubTot_dur_traffic']
            ].std(axis=1)
    )

##select a few of the smallest/most equitable n commute times, extract index
#afCoarseIndex=(
#        set(dfGrid.loc[:,(0,'TotCom')].nsmallest(1).index).union(
#                set(dfGrid.loc[:,(0,'Equity')].nsmallest(1).index)
#                )
#        )

#select a few of the smallest/most equitable n commute times (average, equal weight), extract index
afCoarseIndex=dfGrid.loc[:,[(0,'TotCom'),(0,'Equity')]].mean(axis=1).nsmallest(10).index



#lat/lon
dfGrid.loc[afCoarseIndex,(0,'GMLat')]
dfGrid.loc[afCoarseIndex,(0,'GMLon')]
#afCoarseIndex




#defing highres grid spacing
fHRSpace=spacing[1]/3 # high resolution space (0.33 to avoid overlap)

#define template grid
#xarray centered on origin
afDegTemp=np.linspace(-1*fHRSpace,fHRSpace,num=3,endpoint=True)
aTempGrid=np.array(np.meshgrid(afDegTemp, afDegTemp, indexing='xy')).T.reshape(-1, 2)
dfHRTemplateGrid=pandas.DataFrame(aTempGrid,columns=['RelLon','RelLat']) 

#remove existing data point to avoid duplicating API call
dfHRTemplateGrid.drop(index=[4], inplace=True)

#rotate grid
dfHRTemplateGrid['Lon']=dfHRTemplateGrid['RelLon']*np.cos(fTheta)-dfHRTemplateGrid['RelLat']*np.sin(fTheta)
dfHRTemplateGrid['Lat']=dfHRTemplateGrid['RelLon']*np.sin(fTheta)+dfHRTemplateGrid['RelLat']*np.cos(fTheta)

aHRDFs=[] #ititialize list of highres dataframes
#pass lat/lon along and compute new high-res grid
print('HighRes Lines')
for i in afCoarseIndex:
    print ('i: ',i)
    dfTempGrid=dfHRTemplateGrid.copy()
    
    #correct to final grid
    dfTempGrid['CorrLon']=dfTempGrid['Lon']+dfGrid.loc[i,(0,'GMLon')]
    dfTempGrid['CorrLat']=dfTempGrid['Lat']+dfGrid.loc[i,(0,'GMLat')]

    #plot fine grid
    #plt.plot(dfTempGrid['CorrLon'],dfTempGrid['CorrLat'],
    #         marker='o', color='green', markersize=2, transform=ccrs.Geodetic(),
    #        linestyle='')

    #append temporary data frame to list
    aHRDFs.append(dfTempGrid)
    
    
dfHR=pandas.concat(aHRDFs,ignore_index=True)
#convert to multi index


##############this should be converted to a function
dfHR.columns=indNew

#for each point:
for i in range(len(dfHR)): #whole df
#for i in [3,6]: #whole df
    print ('HRrow i: ',i) #row of initial point
    #for each destination:

    for j in range(len(dfDest)): #whole df
        print ('j: ',j) #row of destination/column of grid
        
        #Compute depart directions
        gmDir_Dtemp = gmaps.directions(
                origin=(dfDest.loc[j,'Lat'],dfDest.loc[j,'Lon']), #dest   lat,lon ,or string array with address
                destination=(dfHR.loc[i,(0,'CorrLat')],dfHR.loc[i,(0,'CorrLon')]), #grid pt   lat,lon ,or string array with address
                 mode='driving',
                 departure_time=dfDest.loc[j,'Depart'],
                 traffic_model="best_guess"
                 )    

        #while loop to catch errors from google maps - perhaps this should be converted into a function
        while (
               'overview_polyline' not in gmDir_Dtemp[0] or
               'duration' not in gmDir_Dtemp[0]['legs'][0] or #dur in seconds
               'duration_in_traffic' not in gmDir_Dtemp[0]['legs'][0] #dur in seconds
               ):
            print('Retrying in 5 seconds')        
            time.sleep(5)
            
            gmDir_Dtemp = gmaps.directions(
                    origin=(dfDest.loc[j,'Lat'],dfDest.loc[j,'Lon']), #dest   lat,lon ,or string array with address
                    destination=(dfGrid.loc[i,(0,'CorrLat')],dfGrid.loc[i,(0,'CorrLon')]), #grid pt   lat,lon ,or string array with address
                     mode='driving',
                     departure_time=dfDest.loc[j,'Depart'],
                     traffic_model="best_guess"
                     )    


        ##store departure columns from leg 0
        #store general depart data
        dfHR.at[i,(j,'D-Polyline')]  =(
                 gmDir_Dtemp[0]['overview_polyline']['points']
                 )#keep as polyline for time
         
        dfHR.at[i,(j,'D-dur')]=(
                 gmDir_Dtemp[0]['legs'][0]['duration']['value'] #dur in seconds
                 )

        dfHR.at[i,(j,'D-dur_traffic')]=(
                 gmDir_Dtemp[0]['legs'][0]['duration_in_traffic']['value'] #dur in seconds
                 )
         
        dfHR.at[i,(j,'D-traffic')]=(
                 dfHR.at[i,(j,'D-dur_traffic')]-dfHR.at[i,(j,'D-dur')]
                 )

        #Compute Arrival directions
        #########Need to readjust to calculate departure time
        #########Maps API will not calculate traffic without a departure time set
        #specified arrival - return trip time(seconds)        
        tdArrivalShift=pandas.Timedelta(
                dfHR.at[i,(j,'D-dur_traffic')] 
                ,unit='s')
        
        #use departure transit time to estimate necessary departure time
        gmDir_Atemp = gmaps.directions(
                origin=(dfHR.loc[i,(0,'CorrLat')],dfHR.loc[i,(0,'CorrLon')]), #grid pt   lat,lon ,or string array with address
                destination=(dfDest.loc[j,'Lat'],dfDest.loc[j,'Lon']), #dest   lat,lon ,or string array with address
                 mode='driving',
        #         #        alternatives=False
        ##        avoid=
        #         units='imperial',
                 departure_time=dfDest.loc[j,'Arrival']-tdArrivalShift,
#                 traffic_model="best_guess"
                 )    
        
        
        #while loop to catch errors from google maps - perhaps this should be converted into a function
        while (
               'overview_polyline' not in gmDir_Atemp[0] or
               'duration' not in gmDir_Atemp[0]['legs'][0] or #dur in seconds
               'duration_in_traffic' not in gmDir_Atemp[0]['legs'][0] #dur in seconds
               ):
            print('Retrying in 5 seconds')        
            time.sleep(5)
            gmDir_Atemp = gmaps.directions(
                    origin=(dfGrid.loc[i,(0,'CorrLat')],dfGrid.loc[i,(0,'CorrLon')]), #grid pt   lat,lon ,or string array with address
                    destination=(dfDest.loc[j,'Lat'],dfDest.loc[j,'Lon']), #dest   lat,lon ,or string array with address
                     mode='driving',
            #         #        alternatives=False
            ##        avoid=
            #         units='imperial',
                     departure_time=dfDest.loc[j,'Arrival']-tdArrivalShift,
    #                 traffic_model="best_guess"
                     )                
        
        
        
        
        ##store arrival columns from leg 0
        if j==0:
            #store start location
             dfHR.at[i,(j,'GMLon')]   =(
                     gmDir_Atemp[0]['legs'][0]['start_location']['lng']
                     )
             dfHR.at[i,(j,'GMLat')]=(
                     gmDir_Atemp[0]['legs'][0]['start_location']['lat']
                     )        

        #store general arrival data
        dfHR.at[i,(j,'A-Polyline')]  =(
                 gmDir_Atemp[0]['overview_polyline']['points']
                 )#keep as polyline for time
         
        dfHR.at[i,(j,'A-dur')]=(
                 gmDir_Atemp[0]['legs'][0]['duration']['value'] #dur in seconds
                 )

        dfHR.at[i,(j,'A-dur_traffic')]=(
                 gmDir_Atemp[0]['legs'][0]['duration_in_traffic']['value'] #dur in seconds
                 )
         
        dfHR.at[i,(j,'A-traffic')]=(
                 dfHR.at[i,(j,'A-dur_traffic')]-dfHR.at[i,(j,'A-dur')]
                 )

        #########Daily Destination Subtotal        
        #total commute time, multiploed by days/week
        dfHR.at[i,(j,'SubTot_dur_traffic')]=(
                 dfHR.at[i,(j,'A-dur_traffic')]+
                 dfHR.at[i,(j,'D-dur_traffic')]#dur in seconds
                 ) * dfDest.loc[j,'NumDays']
        #total extra spent in traffic
        dfHR.at[i,(j,'SubTot_traffic')]=(
                 dfHR.at[i,(j,'A-traffic')]+
                 dfHR.at[i,(j,'D-traffic')]#dur in seconds
                 ) * dfDest.loc[j,'NumDays']
    
#Final point evaluation - all driving for all destinations
#TotCom - total commute time for all destinations
dfHR.loc[:,(0,'TotCom')]=(
    dfHR.loc[
            :, #dataframe row
            pandas.IndexSlice[:, 'SubTot_dur_traffic']
            ].sum(axis=1)
    )
#TotTraf #all extra traffic for all destinations
dfHR.loc[:,(0,'TotTraf')]=(
        dfHR.loc[
        :, #dataframe row
        pandas.IndexSlice[:, 'SubTot_traffic']
        ].sum(axis=1)
    )
#standard deviation to measure equity/fairness of driving
dfHR.loc[:,(0,'Equity')]=(
        dfHR.loc[
            :, #dataframe row
            pandas.IndexSlice[:, 'SubTot_dur_traffic']
            ].std(axis=1)
    )



#Combine all: standard and highres dataframes into main dataframe
dfGrid=pandas.concat([dfGrid,dfHR],ignore_index=True)

#set normalized values for determining color map values
#normval = (value-min) / (max-min)
#dfGrid.loc[:,(0,'Norm')]=(
#        dfGrid.loc[:,(0,'Equity')]-dfGrid.loc[:,(0,'Equity')].min()
#        )/(dfGrid.loc[:,(0,'Equity')].max()-dfGrid.loc[:,(0,'Equity')].min())

#dfGrid.loc[:,(0,'Norm')]=(
#        dfGrid.loc[:,(0,'TotCom')]-dfGrid.loc[:,(0,'TotCom')].min()
#        )/(dfGrid.loc[:,(0,'TotCom')].max()-dfGrid.loc[:,(0,'TotCom')].min())


dfToNorm=dfGrid.loc[:,[(0,'TotCom'),(0,'Equity')]].mean(axis=1)

dfGrid.loc[:,(0,'Norm')]=(
        dfToNorm-dfToNorm.min()
        )/(dfToNorm.max()-dfToNorm.min())




#select direction points for voronoi
points=dfGrid.loc[:,[(0,'GMLon'),(0,'GMLat')]].copy().dropna()



vor = Voronoi(points)
new_regions, new_vertices = voronoi_finite_polygons_2d(vor, 0.75*spacing[1])

#cmap = matplotlib.cm.get_cmap('viridis')
cmap = matplotlib.cm.get_cmap('jet')

dfGrid.loc[:,(0,'Point')]=''
dfGrid.loc[:,(0,'Color')]=''
dfGrid.loc[:,(0,'Vertices')]=''
dfGrid.loc[:,(0,'Shape')]=''

dfGrid.loc[:,(0,'Point')] =dfGrid.loc[:,(0,'Point')].astype(object)
dfGrid.loc[:,(0,'Color')] =dfGrid.loc[:,(0,'Color')].astype(object)
dfGrid.loc[:,(0,'Vertices')]=dfGrid.loc[:,(0,'Vertices')].astype(object)
dfGrid.loc[:,(0,'Shape')] =dfGrid.loc[:,(0,'Vertices')].astype(object)


#for i in range(len(dfGrid)): #Index of the Voronoi region for each input point
for i in range(len(points.index)): #Index of the Voronoi region for each input point
    j=points.index[i] #specify index of the point being evaluated
    #print ('Point Coord: ' + str(vor.points[i]))
    dfGrid.at[j,(0,'Point')]=vor.points[i]
#    print ('Region: '+str(vor.point_region[i]))
#    print ('Region vertices (indices): '+str(new_regions[vor.point_region[i]]))
#    print ('Region vertices (old indices): '+str(vor.regions[vor.point_region[i]]))
    #print ('Region Ridges (indices): '+str(all_ridges[i]))
    aTempVert=new_regions[vor.point_region[i]] #Region vertices (indices)
    dfGrid.at[j,(0,'Vertices')]=new_vertices[aTempVert]
    dfGrid.at[j,(0,'Shape')]=shapely.geometry.Polygon(dfGrid.at[j,(0,'Vertices')])
    #need to properly orient polygons and ensure CCW orientation of points for filled area
    dfGrid.at[j,(0,'Shape')]=shapely.geometry.polygon.orient(dfGrid.at[j,(0,'Shape')], sign=1.0)
    #dfGrid.at[i,'Color']=cmap(dfGrid.loc[i,'Value'],alpha=0.25) #no normalization (no good)
    dfGrid.at[j,(0,'Color')]=cmap(dfGrid.loc[j,(0,'Norm')],alpha=0.35) #Normalization (required)
    
    
    
##############plotting
imagery = OSM() # Use Open street maps data
ax = plt.axes(projection=imagery.crs)
#ax.set_extent([-78, -74, 38, 41], ccrs.Geodetic()) #longitude, latitude (x1,x2,y1,y2)
ax.set_extent(aExtent, ccrs.Geodetic()) #longitude, latitude (x1,x2,y1,y2)

# Add the imagery to the map. Later iterations will need to intellegently determine zoom level
ax.add_image(imagery, 10) #good

#Destinations
plt.plot(dfDest.loc[:,'Lon'], dfDest.loc[:,'Lat'],
         marker='o', color='blue', markersize=9, transform=ccrs.Geodetic(),
         linestyle='')

#plot gm points grid
plt.plot(dfGrid.loc[:,(0,'GMLon')],dfGrid.loc[:,(0,'GMLat')],
         marker='o', color='black', markersize=1, transform=ccrs.Geodetic(),
         linestyle='')

plt.plot(
        aRoute[:,0], #x lng
        aRoute[:,1], #y lat
         marker='o', linestyle='--', color='green', markersize=1, transform=ccrs.Geodetic()
         )    
    
#Plots all the regions and associated colors
ax.add_geometries(np.array(dfGrid.loc[points.index,(0,'Shape')]), ccrs.Geodetic(),
          facecolor=np.array(dfGrid.loc[points.index,(0,'Color')]), edgecolor=None)



