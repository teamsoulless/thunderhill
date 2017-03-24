import math
from geopy.distance import vincenty
from mpl_toolkits.basemap import Basemap

p1 = (-122.34653, 39.53112)
p2 = (-122.33591, 39.5434)

zero = (-122.32115, 39.54733)
ur = zero
ll = p1

mt = Basemap(llcrnrlon=ll[0],llcrnrlat=ll[1],urcrnrlon=ur[0], urcrnrlat=ur[1],
    projection='merc',lon_0=zero[0],lat_0=zero[1],resolution='h', lat_ts=zero[1])

diffx = 1060
diffy = 845

def rescale(simx,simy):
    return 850-simx, 735-simy

def toGPS(simx, simy):
    simx, simy = rescale(int(simx),int(simy))
    projx, projy = simx+diffx, simy + diffy
    lon, lat = mt(projx, projy,inverse=True)
    return [lon, lat]

#http://stackoverflow.com/questions/27461634/calculate-distance-between-a-point-and-a-line-segment-in-latitude-and-longitude
def point_dist(x1,y1,x2,y2):
    return math.sqrt((x1-x2)*(x1-x2) + (y1-y2) * (y1-y2))

def line_dist(pt1, pt2):
    x1,y1,x2,y2 = pt1[0], pt1[1], pt2[0], pt2[1]
    return math.sqrt((x1-x2)*(x1-x2) + (y1-y2) * (y1-y2))

def dist(x1,y1, x2,y2, x3,y3): # x3,y3 is the point
    px = x2-x1
    py = y2-y1

    something = px*px + py*py
    
    if something == 0:
        return math.sqrt((x1-x3)*(x1-x3) + (y1-y3) * (y1-y3))

    u =  ((x3 - x1) * px + (y3 - y1) * py) / float(something)

    if u > 1:
        u = 1
    elif u < 0:
        u = 0

    x = x1 + u * px
    y = y1 + u * py

    dx = x - x3
    dy = y - y3

    dist = math.sqrt(dx*dx + dy*dy)

    return dist

def distance(p0, p1, p2): # p3 is the point
    x0, y0 = p0
    x1, y1 = p1
    x2, y2 = p2
    return dist(x0,y0,x1,y1, x2,y2)

def get_pos(p):
    return (int(p[1][4]), int(p[1][5]))

def get_steering(p):
    return float(p[1][0])

def get_rot_x(p):
    return float(p[1][7])

def get_rot_y(p):
    return float(p[1][8])

def get_rot_z(p):
    return float(p[1][9])

def get_speed(p):
    return float(p[1][3])

