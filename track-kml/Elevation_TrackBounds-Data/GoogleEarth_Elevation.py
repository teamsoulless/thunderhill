import simplejson
import urllib
import numpy as np
import matplotlib.pyplot as plt

from fastkml import  kml
import os


ELEVATION_BASE_URL = 'https://maps.googleapis.com/maps/api/elevation/json'
CHART_BASE_URL = 'https://chart.apis.google.com/chart'

def callElevationAPI(locations,key, **elvtn_args):
    if bool(key):
        elvtn_args.update({'locations': locations,'key': key})
    else:
        elvtn_args.update({'locations': locations})
                              
    url = ELEVATION_BASE_URL + '?' + urllib.urlencode(elvtn_args)
    response = simplejson.load(urllib.urlopen(url))

    # Create a dictionary for each results[] object
    elevationArray = []
    i = 0
    for resultset in response['results']:
        if response['status'] == "OK":
            elevationArray.append(resultset['elevation'])
        else:
            print response['status']

    return elevationArray

def getElevation(data,key):
    numPts = 200
    niters = np.size(data,0)/200
    if np.mod(np.size(data,0),numPts) > 0:
        niters = niters+1
        elevation = []
        for c in range(0,niters):
            str_pts = str(data[c*numPts,1]) + "," + str(data[c*numPts,0])
            if c == niters - 1:
                nPts = min(numPts, np.mod(np.size(data,0),numPts))
            else:
                nPts = numPts
            for c1 in range(1,nPts):
                str_pts = str_pts + "|" + str(data[c*numPts+c1,1]) + "," + str(data[c*numPts+c1,0])
            elev = callElevationAPI(str_pts,key)
            elevation = elevation + elev
    return elevation

def nestedKMLSearch(kmlFolder):
    gps = dict()
    
    if type(kmlFolder) == type(list()):
        for i in range(len(kmlFolder)):
            gps = nestedKMLSearch(kmlFolder[i])
    elif type(kmlFolder) == type(kml.Folder()):
        f = list(kmlFolder.features())
        if type(f[0]) == type(kml.Folder()):
            gps = nestedKMLSearch(f[0])
        else:
            if type(f[0]) == type(kml.Placemark()):
                for i in range(len(f)):
                    gps[f[i].name] = []
                    if "_geometry" in f[i].__dict__.keys():
                        if "geometry" in f[i].__dict__["_geometry"].__dict__.keys():
                            gps[f[i].name] = np.array(f[i].__dict__["_geometry"].__dict__["geometry"])
    else:
        # kmlFolder is a placemark
        gps[kmlFolder.name] = []
        if "_geometry" in kmlFolder.__dict__.keys():
            if "geometry" in kmlFolder.__dict__["_geometry"].__dict__.keys():
                gps[kmlFolder.name] = np.array(kmlFolder.__dict__["_geometry"].__dict__["geometry"])
    return gps

def compareElevation(filename, key):
    doc = file(filename).read()
    k = kml.KML()
    k.from_string(doc)
    features = list(k.features())
    
    Myplaces = list(features[0].features())
    thunderhill = list(Myplaces[0].features())
    gps = []
    for i in thunderhill[0].features():
        gps.append(nestedKMLSearch(i))

    gps_1 = dict()
    for i in range(len(gps)):
        for j in gps[i].keys():
            gps_1[j] = gps[i][j]
            if not (np.max(gps_1[j][:,2]) == 0):
                elevation = getElevation(gps_1[j],key)

                plt.plot(elevation,'b')
                plt.plot(gps_1[j][:,2],'g')
                plt.title("Elevation profile for " + j + " line")
                plt.legend(["Google Elevation API","PolySync"])
                plt.show()

def getCompleteGPSdata(filename, key):
    doc = file(filename).read()
    k = kml.KML()
    k.from_string(doc)
    features = list(k.features())
    
    Myplaces = list(features[0].features())
    thunderhill = list(Myplaces[0].features())
    gps = []
    for i in thunderhill[0].features():
        gps.append(nestedKMLSearch(i))

    gps_1 = dict()
    url = []
    for i in range(len(gps)):
        for j in gps[i].keys():
            gps_1[j] = gps[i][j]
            if (np.max(gps_1[j][:,2]) == 0):
                elevation = getElevation(gps_1[j],key)
                gps_1[j][:,2] = np.array(elevation).reshape(np.size(gps_1[j],0),)

    return gps_1

#if __name__ == '__main__':
#    key = ""
#    cwd = os.path.dirname(os.path.realpath(__file__))
#    filename = os.path.join(cwd,"myplaces.kml")
##    compareElevation(filename, key)
##    print getCompleteGPSdata(filename, key)
