import cv2
import numpy as np
from math import *

def deg2rad(deg):
    return (deg * pi) / 180.0

def rad2deg(rad):
    return (rad * 180.0) / pi

def draw_steering(img, x,y,r,theta):
    theta = -theta
    cv2.circle(img, (x, y), int(r), (255,0,0), 8)
    #cv2.line(img, (x,y), (x+int(r), y), (255,0,0), 8)    

    b=r*np.cos(theta)
    h=b*np.tan(theta)
    xp=x+int(b)
    yp=y-int(h)

    x1 = x - int(b)
    y1=y+int(h)

    cv2.line(img, (x1,y1), (xp,yp), (255,255,255), 8)