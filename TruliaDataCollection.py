# -*- coding: utf-8 -*-
"""
Created on Sun Feb 18 11:40:40 2018

@author: Jason
"""

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
import shapely.prepared
import time

import dill


#start with set of polygons
dfGrid = dill.load(open('jupyterdills/dfGrid.dill', 'rb'))
aRoute = dill.load(open('jupyterdills/aRoute.dill', 'rb'))
dfDest = dill.load(open('jupyterdills/dfDest.dill', 'rb'))
gmDir_jobjob = dill.load(open('jupyterdills/gmDir_jobjob.dill', 'rb'))




import pandas
import urllib3
import re
import json



#import BeautifulSoup4
# or if you're using BeautifulSoup4:
from bs4 import BeautifulSoup



#############For initial polygon:############

#build URL

#Define extent of lat/lon of polygon

i=0

#dfGrid.loc[i,(0,'Vertices')][:,0] #all rows, column 0 (lon)
#dfGrid.loc[i,(0,'Vertices')][:,1] #all rows, column 1 (lat)

fLonMin=dfGrid.loc[i,(0,'Vertices')][:,0].min() #all rows, column 0 (lon)
fLatMin=dfGrid.loc[i,(0,'Vertices')][:,1].min() #all rows, column 1 (lat)

fLonMax=dfGrid.loc[i,(0,'Vertices')][:,0].max() #all rows, column 0 (lon)
fLatmax=dfGrid.loc[i,(0,'Vertices')][:,1].max() #all rows, column 1 (lat)

#https://www.trulia.com/for_sale/39.411985291417,39.807143735765,-75.953610541551,-75.750363471238_xy/
# https://www.trulia.com/for_sale/fLatMin,fLatmax,fLonMin,fLonMax_xy/



#start with first page
iPageNo=1

#valid url:
print('https://www.trulia.com/for_sale/'+str(fLatMin)+','+str(fLatmax)+','+str(fLonMin)+','+str(fLonMax)+'_xy/'+str(iPageNo)+'_p/')

sURL='https://www.trulia.com/for_sale/'+str(fLatMin)+','+str(fLatmax)+','+str(fLonMin)+','+str(fLonMax)+'_xy/'+str(iPageNo)+'_p/'



#####pull data
http = urllib3.PoolManager()
r = http.request('GET', sURL)
markup=r.data





#pagedata to find number of pages
#check number of results
soup = BeautifulSoup(markup, "lxml")
test=soup.find_all(class_=re.compile(r'h6 typeLowlight pbs'))
iTotalResults=int(test[0].string.split()[0])


if iTotalResults>30:
    #only works if >30 results
    test=soup.find_all(class_=re.compile(r'pvl phm'))
    iTotalPages=int(test[-2].string) #second to last element in list of pages, extract string as final page #
else:
    iTotalPages=1


print('total pages: ', iTotalPages)
print('total results: ', iTotalResults)


#Take and store additinal data
#print(re.search(r"DDC\.dataLayer\['vehicles'\]\s*=\s*(.*);", r.data.decode() ,re.DOTALL))#use DOTALL to match newlines
reJSONsearch1=re.search(r"\"cards\":\[(.*?)\],\"currentUrl", r.data.decode(), re.DOTALL)


#append brackets to read into json
dfJson1=pandas.DataFrame(json.loads('['+reJSONsearch1.group(1)+']'))



#set empty dataframe
dfTempHousing=pandas.DataFrame()  
for iPage in range(iTotalPages):
    print('page:',iPage+1)
    
    #build URL
    sURL='https://www.trulia.com/for_sale/'+str(fLatMin)+','+str(fLatmax)+','+str(fLonMin)+','+str(fLonMax)+'_xy/'+str(iPage+1)+'_p/'

    #read data
    http = urllib3.PoolManager()
    r = http.request('GET', sURL)
    
    
    #select data of interest
    reJSONsearch1=re.search(r"\"cards\":\[(.*?)\],\"currentUrl", r.data.decode(), re.DOTALL)
    #append brackets to read into json & append dataframe
    dfTempHousing=dfTempHousing.append(pandas.DataFrame(json.loads('['+reJSONsearch1.group(1)+']')))

dfTempHousing.reset_index(inplace=True)
        



intersect with polygon

#specify polygon
GridPolygon=dfGrid.loc[i,(0,'Shape')]
#prep Polygon
PrepGridPolygon = shapely.prepared.prep(GridPolygon)


#Create array of housing points: need lon,lat format
dfTempHousing[['lat','lon']]=pandas.DataFrame(dfTempHousing.loc[:,'latLng'].values.tolist(), index= dfTempHousing.index)
testpoints=shapely.geometry.MultiPoint(np.array(dfTempHousing.loc[:,['lon','lat']]))

#specify which points fit in poly
hits = map(PrepGridPolygon.contains, testpoints)
dfTempHousing.loc[:,'InPoly']=list(hits)


#drop false, keep true
dfTempHousing=(dfTempHousing.loc[(dfTempHousing.loc[:,'InPoly']==True),:])

#fix values to floats
dfTempHousing.loc[:,'price']=pandas.to_numeric(dfTempHousing.loc[:,'price'].replace({'\$': '', ',': ''}, regex=True))
dfTempHousing.loc[:,'sqft']=pandas.to_numeric(dfTempHousing.loc[:,'sqft'].replace({' sqft': '', ',': ''}, regex=True))
len(dfTempHousing)

#compute basic stats
dfGrid.loc[i,(0,'MedianCost')]=dfTempHousing.loc[:,'price'].median()
dfGrid.loc[i,(0,'Count')]=len(dfTempHousing)

















#All in a for loop:


#setup blank housing dataframe
dfHousing=pandas.DataFrame()

for i in range(len(dfGrid)):
    print('polygon: ',i)
    #Define extent of lat/lon of polygon
    
    fLonMin=dfGrid.loc[i,(0,'Vertices')][:,0].min() #all rows, column 0 (lon)
    fLatMin=dfGrid.loc[i,(0,'Vertices')][:,1].min() #all rows, column 1 (lat)
    
    fLonMax=dfGrid.loc[i,(0,'Vertices')][:,0].max() #all rows, column 0 (lon)
    fLatMax=dfGrid.loc[i,(0,'Vertices')][:,1].max() #all rows, column 1 (lat)
    
    #https://www.trulia.com/for_sale/39.411985291417,39.807143735765,-75.953610541551,-75.750363471238_xy/
    # https://www.trulia.com/for_sale/fLatMin,fLatmax,fLonMin,fLonMax_xy/
    
    
    
    #start with first page
    iPageNo=1
    
    #valid url:
    print('https://www.trulia.com/for_sale/'+str(fLatMin)+','+str(fLatMax)+','+str(fLonMin)+','+str(fLonMax)+'_xy/'+str(iPageNo)+'_p/')
    
    sURL='https://www.trulia.com/for_sale/'+str(fLatMin)+','+str(fLatMax)+','+str(fLonMin)+','+str(fLonMax)+'_xy/'+str(iPageNo)+'_p/'
    
    
    
    #####pull data
    http = urllib3.PoolManager()
    r = http.request('GET', sURL)
    markup=r.data
    
    
    
    
    
    #pagedata to find number of pages
    #check number of results
    soup = BeautifulSoup(markup, "lxml")
    test=soup.find_all(class_=re.compile(r'h6 typeLowlight pbs'))
    iTotalResults=int(test[0].string.split()[0])
    
    
    if iTotalResults>30:
        #only works if >30 results
        test=soup.find_all(class_=re.compile(r'pvl phm'))
        iTotalPages=int(test[-2].string) #second to last element in list of pages, extract string as final page #
    else:
        iTotalPages=1
    
    
    print('total pages: ', iTotalPages)
    print('total results: ', iTotalResults)
    
    
    #Take and store additinal data
    #print(re.search(r"DDC\.dataLayer\['vehicles'\]\s*=\s*(.*);", r.data.decode() ,re.DOTALL))#use DOTALL to match newlines
    reJSONsearch1=re.search(r"\"cards\":\[(.*?)\],\"currentUrl", r.data.decode(), re.DOTALL)
    
    
    #append brackets to read into json
    dfJson1=pandas.DataFrame(json.loads('['+reJSONsearch1.group(1)+']'))
    
    ###
    #debug limit
    if iTotalPages >3:
        iTotalPages=3
        print('Limiting Number of Pages')
    ###
    
    #set empty dataframe
    dfTempHousing=pandas.DataFrame()  
    for iPage in range(iTotalPages):
        print('page:',iPage+1)
        
        #build URL
        sURL='https://www.trulia.com/for_sale/'+str(fLatMin)+','+str(fLatmax)+','+str(fLonMin)+','+str(fLonMax)+'_xy/'+str(iPage+1)+'_p/'
    
        #read data
        http = urllib3.PoolManager()
        r = http.request('GET', sURL)
        
        
        #select data of interest
        reJSONsearch1=re.search(r"\"cards\":\[(.*?)\],\"currentUrl", r.data.decode(), re.DOTALL)
        #append brackets to read into json & append dataframe
        dfTempHousing=dfTempHousing.append(pandas.DataFrame(json.loads('['+reJSONsearch1.group(1)+']')))
    
    dfTempHousing.reset_index(inplace=True)
            
    
    
    
    #intersect with polygon
    
    #specify polygon
    GridPolygon=dfGrid.loc[i,(0,'Shape')]
    #prep Polygon
    PrepGridPolygon = shapely.prepared.prep(GridPolygon)
    
    
    #Create array of housing points: need lon,lat format
    dfTempHousing[['lat','lon']]=pandas.DataFrame(dfTempHousing.loc[:,'latLng'].values.tolist(), index= dfTempHousing.index)
    testpoints=shapely.geometry.MultiPoint(np.array(dfTempHousing.loc[:,['lon','lat']]))
    
    #specify which points fit in poly
    hits = map(PrepGridPolygon.contains, testpoints)
    dfTempHousing.loc[:,'InPoly']=list(hits)
    
    
    #drop false, keep true
    dfTempHousing=(dfTempHousing.loc[(dfTempHousing.loc[:,'InPoly']==True),:])
    
    #fix values to floats
    dfTempHousing.loc[:,'price']=pandas.to_numeric(dfTempHousing.loc[:,'price'].replace({'\$': '', ',': ''}, regex=True))
    dfTempHousing.loc[:,'sqft']=pandas.to_numeric(dfTempHousing.loc[:,'sqft'].replace({' sqft': '', ',': ''}, regex=True))
    len(dfTempHousing)
    
    #compute basic stats
    dfGrid.loc[i,(0,'MedianCost')]=dfTempHousing.loc[:,'price'].median()
    dfGrid.loc[i,(0,'Count')]=len(dfTempHousing)

    #add to full housing dataframe for later
    dfHousing=dfHousing.append(dfTempHousing)









##############plotting Test
aExtent=[
        fLonMin,
        fLonMax,
        fLatMin,
        fLatMax,
        ]

imagery = OSM() # Use Open street maps data
ax = plt.axes(projection=imagery.crs)
#ax.set_extent([-78, -74, 38, 41], ccrs.Geodetic()) #longitude, latitude (x1,x2,y1,y2)
ax.set_extent(aExtent, ccrs.Geodetic()) #longitude, latitude (x1,x2,y1,y2)

# Add the imagery to the map. Later iterations will need to intellegently determine zoom level
ax.add_image(imagery, 7) #good

#plots region of interest
x,y = GridPolygon.exterior.xy
ax.plot(
        x, #x lng
        y, #y lat
         marker='o', linestyle='--', color='green', markersize=1, transform=ccrs.Geodetic()
         )   
    
#plots housing points 
ax.plot(
        dfTempHousing.loc[:,'lon'], #x lng
        dfTempHousing.loc[:,'lat'], #y lat
         marker='o', linestyle='', color='blue', markersize=1, transform=ccrs.Geodetic()
         )   



















































dfGrid.loc[i,(0,'Shape')]

dfGrid.loc[i,(0,'Shape')]

np.array(dfGrid.loc[i,(0,['CorrLat','CorrLon'])])

dfTempHousing.loc[0,'latLng']


dfGrid.loc[i,(0,'Shape')].contains(shapely.geometry.Point([-77.3956, 38.82089])) #need to do lon, lat


dfGrid.loc[i,(0,'Shape')].contains(shapely.geometry.Point(np.array(dfGrid.loc[:,(0,['CorrLon','CorrLat'])]))) #need to do lon, lat


#list of coords
np.array(dfGrid.loc[:,(0,['CorrLon','CorrLat'])])
#convert to list of points
shapely.geometry.Point(np.array(dfGrid.loc[:,(0,['CorrLon','CorrLat'])]))
shapely.geometry.Point((-76.75      ,  39.75      ),       (-76.5       ,  40.5       ))
shapely.geometry.Point([-76.75      ,  39.75      ],       [-76.5       ,  40.5       ])
shapely.geometry.MultiPoint([(-76.75      ,  39.75      ),       (-76.5       ,  40.5       )])
test=shapely.geometry.MultiPoint([[-76.75      ,  39.75      ],       [-76.5       ,  40.5       ]])
test[0]
testpoints=shapely.geometry.MultiPoint(np.array(dfGrid.loc[:,(0,['CorrLon','CorrLat'])]))

#prep shape
prepared_polygon = shapely.prepared.prep(dfGrid.loc[i,(0,'Shape')])

#look for intersection
hits = map(prepared_polygon.contains, testpoints)
print (hits)
len(map(prepared_polygon.contains, testpoints))
hits = map(prepared_polygon.contains, [testpoints[0],testpoints[1]])
len(map(prepared_polygon.contains, [testpoints[0],testpoints[1]]))
list (hits)
prepared_polygon.contains(testpoints[0],testpoints[1])







add metrics to dfGrid
add to full housing dataframe for later


shapely.geometry.Point([38.82089, -77.3956])














#for case where only one page
if int(dPageInfo['vehicleResultCount'])<=int(dPageInfo['vehicleCountPerPage']):
    print ('Execute '+sURL)
    #################################
    r = http.request('GET', sURL)
    markup=r.data
    #soup = BeautifulSoup(markup, "html5lib")
    soup = BeautifulSoup(markup, "lxml")
    test=soup.find_all(class_=re.compile(r'hproduct auto .*'))
    
    #Take and store additinal data
    #print(re.search(r"DDC\.dataLayer\['vehicles'\]\s*=\s*(.*);", r.data.decode() ,re.DOTALL))#use DOTALL to match newlines
    reJSONsearch=re.search(r"DDC\.dataLayer\['vehicles'\]\s*=\s*(.*?)];", r.data.decode(), re.DOTALL)
    
    try:
        reJSONsub=re.sub(r'[\n\t]|//getting vehicle info and setting null values','',reJSONsearch.group(1))    
        reJSONsub=reJSONsub+']' if  reJSONsub[-1]!=']' else print ('OK')
        dfJson=pandas.DataFrame(json.loads(reJSONsub))
        adfAllData2=adfAllData2.append(pandas.DataFrame(json.loads(reJSONsub)))
    #searchbox_result = re.match("^.*(?=(\())", searchbox.group()
    except:
        break
        #searchbox_result = None
        
    
    ##Iterate over all cars from dealer
    for i in range(len(test)):
        print (i,j,k)
   
    #####take attributes 
        testdict=test[i].attrs #lists attributes -- take these and append to dataframe
        del testdict['class']
      
        #take description 
        descriptiontest=test[i].find_all(class_='description')
        descriptiontest[0].find_all('dt') #make into columns
        descriptiontest[0].find_all('dd') #make into data
    
        asDescCol=[]
        for dt in descriptiontest[0].find_all('dt'):
            asDescCol.append(str(dt.contents[0]))#make into columns
        asDescData=[]
        for dd in descriptiontest[0].find_all('dd'): #make into data
            asDescData.append(str(dd.contents[0]))#make into columns
            
        #take value 
        valuetest=test[i].find_all(class_=re.compile(r'value|h. price'))
        if len(valuetest)>0:       
            fValue=(re.sub("[^0-9.]", "", valuetest[-1].get_text()))
        else:
            fValue=None 
            
        #take url
        #urltest=test[i].find_all(class_='url')
        
        #Store values through joining of dataframes
        #http://pandas.pydata.org/pandas-docs/stable/merging.html
        adfAllData=adfAllData.append(
             pandas.DataFrame(
            {
            **testdict,
            **{'Dealer':sDealer,'Value':fValue},
            **dict(zip(asDescCol,asDescData))
            },
            index=[k])
        )

        #increment on car
        k+=1
    #################################

else: #for multipage case
    while int(dPageInfo['vehicleResultCount'])>int(dPageInfo['vehicleCountPerPage'])*(int(dPageInfo['vehicleCurrentPage'])):
        print ('Execute '+sURL)
        #################################
        r = http.request('GET', sURL)
        markup=r.data
        #soup = BeautifulSoup(markup, "html5lib")
        soup = BeautifulSoup(markup, "lxml")
        test=soup.find_all(class_=re.compile(r'hproduct auto .*'))
                        
        #Take and store additinal data
        #print(re.search(r"DDC\.dataLayer\['vehicles'\]\s*=\s*(.*);", r.data.decode() ,re.DOTALL))#use DOTALL to match newlines
        reJSONsearch=re.search(r"DDC\.dataLayer\['vehicles'\]\s*=\s*(.*?)];", r.data.decode(), re.DOTALL)
        try:            
            reJSONsub=re.sub(r'[\n\t]|//getting vehicle info and setting null values','',reJSONsearch.group(1))    
            reJSONsub=reJSONsub+']' if  reJSONsub[-1]!=']' else print ('OK')
            dfJson=pandas.DataFrame(json.loads(reJSONsub))
            adfAllData2=adfAllData2.append(pandas.DataFrame(json.loads(reJSONsub)))
        except:
            break
        
        ##Iterate over all cars from dealer
        for i in range(len(test)):
            print (i,j,k)
       
        #####take attributes 
            testdict=test[i].attrs #lists attributes -- take these and append to dataframe
            
            del testdict['class']
          
            #take description 
            descriptiontest=test[i].find_all(class_='description')
            descriptiontest[0].find_all('dt') #make into columns
            descriptiontest[0].find_all('dd') #make into data
        
            asDescCol=[]
            for dt in descriptiontest[0].find_all('dt'):
                asDescCol.append(str(dt.contents[0]))#make into columns
            asDescData=[]
            for dd in descriptiontest[0].find_all('dd'): #make into data
                asDescData.append(str(dd.contents[0]))#make into columns
                
            #take value 
            valuetest=test[i].find_all(class_=re.compile(r'value|h. price'))
            if len(valuetest)>0:       
                fValue=(re.sub("[^0-9.]", "", valuetest[-1].get_text()))
            else:
                fValue=None 
            
            #take url
            #urltest=test[i].find_all(class_='url')

            
            #Store values through joining of dataframes
            #http://pandas.pydata.org/pandas-docs/stable/merging.html
            adfAllData=adfAllData.append(
                 pandas.DataFrame(
                {
                **testdict,
                **{'Dealer':sDealer,'Value':fValue},
                **dict(zip(asDescCol,asDescData))
                },
                index=[k])
            )
    
            #increment on car
            k+=1

    
        #################################
        r = http.request('GET', sURL)
        markup=r.data
        reData=re.search(r"DDC\.dataLayer\.page\.attributes\s*=\s*.*// getting vehicle page results(.*?)};", r.data.decode(), re.DOTALL)
        reData2=re.sub(r'[\n\t]','',reData.group(1))    
        dPageInfo=dict(eval('{'+reData2+'}'))
        #sURL1=sURL+'&start='+str(int(dPageInfo['vehicleCountPerPage'])*int(dPageInfo['vehicleCurrentPage']))
        sStart=str(int(dPageInfo['vehicleCountPerPage'])*int(dPageInfo['vehicleCurrentPage']))
        sURL=re.sub(r'start=.*?&',(r'start='+sStart+'&'),sURL)        
        #print (sURL)

#increment on dealer
j+=1
    
####merge dataframes and Save data
#adfAllData.rename(columns={'data-vin': 'vin'}, inplace=True)
#dfconcat=pandas.DataFrame.merge(adfAllData,adfAllData2,on='vin')
#dfconcat['Date']=pandas.to_datetime('now')
#  
#dfconcat.to_csv('output_all.csv')

#adfAllData.rename(columns={'data-vin': 'vin'}, inplace=True)
concat1=adfAllData.rename(columns={'data-vin': 'vin'}).drop_duplicates('vin',keep='first')
concat2=adfAllData2.drop_duplicates('vin',keep='first')
dfconcat=pandas.DataFrame.merge(concat1,concat2,on='vin')
dfconcat['Date']=pandas.to_datetime('now',utc=True).tz_convert('US/Eastern')
  
dfconcat.to_csv('output_all_'+str(pandas.to_datetime('today'))[0:10]+'.csv')