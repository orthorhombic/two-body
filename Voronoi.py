# -*- coding: utf-8 -*-
"""
Created on Fri Dec 29 21:35:31 2017

@author: Jason
"""

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
                print ('continue',  (p2, v1, v2))
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




imagery = OSM() # Use Open street maps data
ax = plt.axes(projection=imagery.crs)
ax.set_extent([-78, -74, 38, 41], ccrs.Geodetic()) #longitude, latitude (x1,x2,y1,y2)


# Add the imagery to the map. Later iterations will need to intellegently determine zoom level
#ax.add_image(imagery, 5) #too low resolution
ax.add_image(imagery, 7) #good
#ax.add_image(imagery, 8) #too high resolution

#These are values which will be defined through address lookup in other modules
afDest1=np.array([-77.0,39.0]) #lon/lat #make sure this is a float
afDest2=np.array([-75.0,40.0]) #lon/lat


#Start
plt.plot(afDest1[0], afDest1[1],
         marker='o', color='blue', markersize=9, transform=ccrs.Geodetic(),
         linestyle='')

#Stop
plt.plot(afDest2[0], afDest2[1],
         marker='o', color='red', markersize=9, transform=ccrs.Geodetic(),
         linestyle='')

##Define square (not rotated)

origin=np.array([0,0])
corner=afDest2-afDest1
#pythagorean theorm for distance, 45-45-90 triangle for ratio to side
hypotenuse=np.sqrt((corner[0]**2+corner[1]**2)) 
side=hypotenuse/np.sqrt(2) 
#specify spacing for side
spacing=np.linspace(0,side,num=10,endpoint=True)

#Create an initial grid 
 #Create a mesh grid
 #Transpose to make each lat/lon point into a row
 #reshape to combine multiple arrays into one
aInitGrid=np.array(np.meshgrid(spacing, spacing, indexing='xy')).T.reshape(-1, 2)


#test= np.array(np.meshgrid([0,1,2], [4,5,6], indexing='ij'))
#test=np.array(np.meshgrid([0,1,2], [4,5,6], indexing='xy')).T.reshape(-1, 2)



# dataframe of (lon/lat) pairs
dfGrid=pandas.DataFrame(aInitGrid,columns=['RelLon','RelLat']) 


##do the math to rotate the grid and reset origin

#calculate rotation of square relative to lat/lon, subtract 45deg inside square
fTheta=np.arcsin(corner[1]/hypotenuse)-(np.pi/4) #use to avoid a divide by zero possibility



#rotate grid
dfGrid['Lon']=dfGrid['RelLon']*np.cos(fTheta)-dfGrid['RelLat']*np.sin(fTheta)
dfGrid['Lat']=dfGrid['RelLon']*np.sin(fTheta)+dfGrid['RelLat']*np.cos(fTheta)

#move back to origin
dfGrid['CorrLon']=dfGrid['Lon']+afDest1[0]
dfGrid['CorrLat']=dfGrid['Lat']+afDest1[1]




#plot coarse grid
plt.plot(dfGrid['CorrLon'],dfGrid['CorrLat'],
         marker='o', color='black', markersize=1, transform=ccrs.Geodetic(),
         linestyle='')



#for arbitrary point, increase resolution 
aHRDFs=[] #ititialize list of highres dataframes
for i in [25,35,26,36, 64]:
    dfGrid.loc[i,'RelLon']
    spacing[1] # default space
    fHRSpace=spacing[1]/3 # high resolution space (0.33 to avoid overlap)
    
    #xarray centered on coarse point
    fLonTemp=dfGrid.loc[i,'RelLon']
    afLonTemp=np.linspace(fLonTemp-fHRSpace,fLonTemp+fHRSpace,num=3,endpoint=True)
    #yarray centered on coarse point
    fLatTemp=dfGrid.loc[i,'RelLat']
    afLatTemp=np.linspace(fLatTemp-fHRSpace,fLatTemp+fHRSpace,num=3,endpoint=True)
    
    aTempGrid=np.array(np.meshgrid(afLonTemp, afLatTemp, indexing='xy')).T.reshape(-1, 2)
    dfTempGrid=pandas.DataFrame(aTempGrid,columns=['RelLon','RelLat']) 
    
    #remove existing data point to avoid duplicating API call
    dfTempGrid.drop(index=[4], inplace=True)
    
    
    #rotate grid
    dfTempGrid['Lon']=dfTempGrid['RelLon']*np.cos(fTheta)-dfTempGrid['RelLat']*np.sin(fTheta)
    dfTempGrid['Lat']=dfTempGrid['RelLon']*np.sin(fTheta)+dfTempGrid['RelLat']*np.cos(fTheta)
    
    
    #correct to final grid
    dfTempGrid['CorrLon']=dfTempGrid['Lon']+afDest1[0]
    dfTempGrid['CorrLat']=dfTempGrid['Lat']+afDest1[1]



    #plot fine grid
    plt.plot(dfTempGrid['CorrLon'],dfTempGrid['CorrLat'],
             marker='o', color='green', markersize=2, transform=ccrs.Geodetic(),
             linestyle='')

    #append temporary data frame to list
    aHRDFs.append(dfTempGrid)



#Combine all: append temp dataframes with main dataframe
#temp=dfGrid.copy()
aHRDFs.append(dfGrid) #add main dataframe
dfGrid=pandas.concat(aHRDFs,ignore_index=True) #combine everything


#figure out how to plot points in dataframe to voronoi plot
#done to most acurately visualize transit time near points
from scipy.spatial import Voronoi, voronoi_plot_2d
import shapely.geometry
import shapely.ops



#points = np.random.random((10, 2))
points=dfGrid.loc[:,['CorrLon','CorrLat']]


vor = Voronoi(points)
#voronoi_plot_2d(vor)

#assign random values to dataframe column
dfGrid.loc[:,'Value']=np.random.random((len(dfGrid),1))
dfGrid.loc[:,'Value']=dfGrid.loc[:,'Value']*20 # to test if normalization is necessary

#colorize values on scale


#dfGrid.loc[:,'Value'].max()
#dfGrid.loc[:,'Value'].min()
#dfGrid.loc[:,'Norm'].max()
#dfGrid.loc[:,'Norm'].min()

#set normalized values for determining color map values
#normval = (value-min) / (max-min)
dfGrid.loc[:,'Norm']=(dfGrid.loc[:,'Value']-dfGrid.loc[:,'Value'].min())/(dfGrid.loc[:,'Value'].max()-dfGrid.loc[:,'Value'].min())


points=dfGrid.loc[:,['CorrLon','CorrLat']]


vor = Voronoi(points)


new_regions, new_vertices = voronoi_finite_polygons_2d(vor, spacing[1])


cmap = matplotlib.cm.get_cmap('Spectral')

#use data frame to reference polygon from each point

dfGrid.loc[:,'Point']=''
dfGrid.loc[:,'Color']=''
dfGrid.loc[:,'Vertices']=''
dfGrid.loc[:,'Shape']=''

dfGrid.loc[:,'Point']=dfGrid.loc[:,'Point'].astype(object)
dfGrid.loc[:,'Color']=dfGrid.loc[:,'Color'].astype(object)
dfGrid.loc[:,'Vertices']=dfGrid.loc[:,'Vertices'].astype(object)
dfGrid.loc[:,'Shape']=dfGrid.loc[:,'Vertices'].astype(object)

for i in range(len(dfGrid)): #Index of the Voronoi region for each input point
    #print ('Point Coord: ' + str(vor.points[i]))
    dfGrid.at[i,'Point']=vor.points[i]
#    print ('Region: '+str(vor.point_region[i]))
#    print ('Region vertices (indices): '+str(new_regions[vor.point_region[i]]))
#    print ('Region vertices (old indices): '+str(vor.regions[vor.point_region[i]]))
    #print ('Region Ridges (indices): '+str(all_ridges[i]))
    aTempVert=new_regions[vor.point_region[i]] #Region vertices (indices)
    dfGrid.at[i,'Vertices']=new_vertices[aTempVert]
    dfGrid.at[i,'Shape']=shapely.geometry.Polygon(dfGrid.at[i,'Vertices'])
    #need to properly orient polygons and ensure CCW orientation of points for filled area
    dfGrid.at[i,'Shape']=shapely.geometry.polygon.orient(dfGrid.at[i,'Shape'], sign=1.0)
    dfGrid.at[i,'Color']=cmap(dfGrid.loc[i,'Value'],alpha=0.25)



#for i in range(len(dfGrid)):
#    print(i)
#    ax.add_geometries([dfGrid.at[i,'Shape']], ccrs.Geodetic(),
#              facecolor=[0,0,0,0], edgecolor='black')
#
#
#for i in [40]:
#    print(i)
#    ax.add_geometries([dfGrid.at[i,'Shape']], crs=ccrs.Mercator,
#              facecolor=[0,0,0,0], edgecolor='red')
#
#for i in [45]:
#    print(i)
#    ax.add_geometries([dfGrid.at[i,'Shape']], crs=ccrs.PlateCarree(),
#              facecolor=[0,0,0,0], edgecolor='red')

    ax.add_geometries(np.array(dfGrid.loc[:,'Shape']), ccrs.Geodetic(),
              facecolor=np.array(dfGrid.loc[:,'Color']), edgecolor=None)


i=0
print (i)
ax.add_geometries([dfGrid.loc[i,'Shape']], ccrs.Geodetic(),
          facecolor=[dfGrid.loc[i,'Color']], edgecolor=None)
i+=1

#broken
i=14 

i=0
print (i)
dfGrid.loc[i,'Shape']
i+=1

i=1
dfGrid.at[i,'Vertices']
shapely.geometry.Polygon
poly=shapely.geometry.Polygon(dfGrid.at[i,'Vertices'])
list(shapely.geometry.Polygon(dfGrid.at[i,'Vertices']).exterior.coords)
list(shapely.geometry.Polygon(dfGrid.at[i,'Vertices']).interiors)
print(shapely.geometry.polygon.orient(poly, sign=1.0))
