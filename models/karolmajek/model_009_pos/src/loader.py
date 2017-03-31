import numpy as np
import pandas as pd
import cv2
from glob import glob
import os
import csv

# Sklearn
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from transformations import Preproc, RandomShift, RandomFlip, RandomBrightness, speedToClass
from config import *

show_images=True
show_images=False


import time
import utm
from geopy.distance import vincenty

ll = (-122.34306, 39.53284)   # 37M 559275mE 9848303mN
ur = (-122.33776, 39.54123)   # 37M 560209mE 9848312mN

# Generated from: nodejs GPSLatLong2UTM.js
UTM_path = [
    [556819.823369298, 4376630.074165889],
    [556805.650790522, 4376742.486302503],
    [556802.1238310878, 4376752.327299194],
    [556797.004905037, 4376758.949030364],
    [556790.7445986632, 4376764.186122571],
    [556782.1777714695, 4376768.7181542525],
    [556775.4619218896, 4376772.797630742],
    [556770.3515047785, 4376778.265158381],
    [556766.3925052953, 4376783.752239646],
    [556765.2140497837, 4376787.41731796],
    [556746.6380893289, 4376822.064872666],
    [556742.5447797973, 4376830.636468087],
    [556741.9244143113, 4376836.714101026],
    [556742.1040275982, 4376842.68661478],
    [556743.1092729353, 4376847.399914415],
    [556746.0619463512, 4376853.736840777],
    [556748.683388908, 4376857.196728521],
    [556761.5352933934, 4376881.175832345],
    [556765.2905187453, 4376887.052504549],
    [556769.7560368717, 4376890.992089035],
    [556775.2622117667, 4376894.817228478],
    [556781.4625108498, 4376897.737360328],
    [556787.1005962924, 4376898.810954876],
    [556792.5169713517, 4376899.649848904],
    [556814.171765087, 4376900.952091674],
    [556825.9152920799, 4376902.64771292],
    [556833.9656089155, 4376905.237416394],
    [556843.8337940382, 4376910.704001542],
    [556852.3134213871, 4376916.848522062],
    [556860.2090013095, 4376925.286226419],
    [556864.8870395265, 4376931.868963813],
    [556869.088476273, 4376940.157427281],
    [556871.9063425458, 4376948.446817077],
    [556873.5660433229, 4376957.637798812],
    [556872.9059148362, 4376969.109188728],
    [556871.5718145098, 4376978.744306643],
    [556867.810959091, 4376987.673448086],
    [556864.0714360261, 4376994.871322429],
    [556859.303022026, 4377000.5854819575],
    [556855.2408456483, 4377004.917476291],
    [556851.1821005888, 4377008.783344851],
    [556843.9947298435, 4377013.902588822],
    [556836.1275991215, 4377017.973548311],
    [556827.5814403858, 4377020.896340846],
    [556816.7338939247, 4377022.425972237],
    [556806.1301831241, 4377022.348072044],
    [556796.9252971965, 4377021.359252923],
    [556789.0986243471, 4377019.925511599],
    [556781.063570797, 4377015.27157263],
    [556573.7292688775, 4376874.40561688],
    [556565.2416781799, 4376868.139261163],
    [556558.5981885456, 4376862.352554217],
    [556557.0483310376, 4376854.538712078],
    [556557.1087982607, 4376846.270494228],
    [556557.1675609464, 4376838.235340337],
    [556555.3921837813, 4376828.355459458],
    [556552.2268787316, 4376820.518706539],
    [556547.2171238372, 4376811.070231053],
    [556541.0644607875, 4376801.613403257],
    [556531.6943581767, 4376791.4449312845],
    [556523.90934363, 4376784.27365231],
    [556515.6704181702, 4376775.711705682],
    [556509.1689304269, 4376766.940485829],
    [556505.7468858059, 4376761.299454233],
    [556502.9042310822, 4376756.339688323],
    [556499.8429328257, 4376749.5470127985],
    [556497.6894335148, 4376744.359207488],
    [556495.9982652441, 4376739.407856463],
    [556494.6466100061, 4376735.0361271575],
    [556493.0755003551, 4376728.94247044],
    [556492.2054705946, 4376722.165805311],
    [556491.4353973819, 4376716.999206436],
    [556490.7831972906, 4376712.166434735],
    [556490.2543825444, 4376705.736327338],
    [556490.3030211494, 4376699.07736165],
    [556490.5658446911, 4376693.685231451],
    [556491.3080298626, 4376687.375396654],
    [556492.266914263, 4376681.988350871],
    [556494.1614199451, 4376675.5759448605],
    [556495.6993857024, 4376669.738075969],
    [556497.937712214, 4376663.317083599],
    [556500.3834163146, 4376659.095182666],
    [556504.6853705766, 4376654.87574512],
    [556510.5968031748, 4376650.324004872],
    [556517.7622229807, 4376647.046703596],
    [556523.4206405458, 4376645.367730853],
    [556531.4956595218, 4376644.627631856],
    [556538.1913655276, 4376643.300316806],
    [556543.5077593034, 4376641.385782115],
    [556550.0957410464, 4376638.337365073],
    [556556.0029932288, 4376634.362769542],
    [556621.1226717028, 4376586.847721664],
    [556625.41272605, 4376583.094415237],
    [556629.5944826928, 4376578.874165656],
    [556633.8946226767, 4376573.744677827],
    [556637.7427283516, 4376568.145729547],
    [556640.3164867894, 4376562.881515558],
    [556642.0874201854, 4376556.934391812],
    [556644.2393811638, 4376547.072158077],
    [556645.2261046184, 4376537.900619624],
    [556644.5939186545, 4376529.172278329],
    [556642.5011432761, 4376516.870503548],
    [556640.9426985707, 4376509.056585676],
    [556639.610424291, 4376502.043443069],
    [556639.1867659474, 4376496.524202927],
    [556639.7037260353, 4376489.302625273],
    [556640.8066145477, 4376480.697980016],
    [556642.5894319536, 4376473.130509078],
    [556652.0281048274, 4376443.465770014],
    [556653.0828299529, 4376440.265922629],
    [556654.2755882231, 4376434.647533262],
    [556655.156551918, 4376424.665005215],
    [556655.350027005, 4376414.677441572],
    [556654.3866934374, 4376404.226351101],
    [556652.4904231104, 4376395.599740706],
    [556649.5601676378, 4376386.166439493],
    [556646.408176098, 4376376.498439539],
    [556617.5261113667, 4376306.841371429],
    [556615.8399530456, 4376301.20190502],
    [556613.9326455214, 4376294.0735726],
    [556612.3682729696, 4376287.058739383],
    [556611.6217850288, 4376278.6736334665],
    [556611.6747452599, 4376271.437561792],
    [556612.1984312234, 4376263.294828581],
    [556613.6358070546, 4376255.957901033],
    [556616.113703918, 4376247.363319298],
    [556619.4063136274, 4376239.007778029],
    [556622.6810748114, 4376231.917377922],
    [556625.6039453042, 4376227.110768979],
    [556633.0429528294, 4376218.219546603],
    [556641.0485826668, 4376210.586651885],
    [556649.6352787388, 4376203.41307418],
    [556661.2067244762, 4376197.071586001],
    [556676.9223657108, 4376191.67058579],
    [556690.5380557012, 4376189.01784376],
    [556704.1607221728, 4376186.587153972],
    [556716.174419276, 4376183.234583609],
    [556727.0364160506, 4376178.719303533],
    [556734.2168176657, 4376174.643187214],
    [556742.7772991644, 4376169.889071413],
    [556751.8153790819, 4376163.296057259],
    [556761.0872204971, 4376156.4716926],
    [556769.1956817285, 4376151.24813625],
    [556778.6826447933, 4376146.7339351745],
    [556788.38475071, 4376143.353409377],
    [556799.699103058, 4376140.450886419],
    [556809.6299566904, 4376138.6925011985],
    [556819.0831754148, 4376137.60765109],
    [556828.5365113092, 4376137.677095531],
    [556834.9803685388, 4376139.111797304],
    [556868.1129967818, 4376148.300997177],
    [556873.727140218, 4376152.704128489],
    [556876.3343281504, 4376158.1284473725],
    [556877.2083417252, 4376164.328042732],
    [556874.5132715993, 4376170.856559011],
    [556868.6013422308, 4376175.396925701],
    [556863.2828333359, 4376177.544303506],
    [556825.48893115, 4376192.194541831],
    [556820.0459741531, 4376194.907077011],
    [556811.9478282619, 4376199.897579757],
    [556803.4821852093, 4376206.949781476],
    [556799.4160940523, 4376210.593648152],
    [556721.0327277103, 4376290.718330059],
    [556717.4248418615, 4376295.175816185],
    [556714.1582693012, 4376299.968772539],
    [556710.42858297, 4376305.8016282795],
    [556707.1537702349, 4376312.8919925755],
    [556704.9194871916, 4376318.724715517],
    [556702.6640540012, 4376327.442989514],
    [556701.1140730372, 4376334.890065835],
    [556700.1389787215, 4376342.463444236],
    [556700.208635789, 4376348.202069332],
    [556700.1690965947, 4376353.595829118],
    [556700.934842537, 4376359.339557077],
    [556715.6237388911, 4376447.039523411],
    [556716.7508660316, 4376450.377449357],
    [556718.6824527478, 4376454.176328366],
    [556722.7968080112, 4376457.991215417],
    [556728.1982774397, 4376460.905437533],
    [556733.141423593, 4376461.862900476],
    [556738.7994189215, 4376461.438251746],
    [556744.4622561141, 4376459.182329016],
    [556749.5560036578, 4376456.012127454],
    [556752.9283884075, 4376452.01908076],
    [556755.4989235783, 4376447.209933989],
    [556766.6914457628, 4376414.006558802],
    [556772.1736996055, 4376404.745943168],
    [556776.9431375248, 4376398.931847487],
    [556781.8150934667, 4376394.37267963],
    [556788.5414554606, 4376388.905923452],
    [556794.5556857836, 4376385.620421416],
    [556826.92710161, 4376373.116682523],
    [556836.9473326703, 4376373.190307461],
    [556845.2294152384, 4376375.892702398],
    [556849.457292943, 4376380.6296974085],
    [556850.3059291652, 4376390.280854663],
]

#calculate new way points and initialize distance between way points
total_lap_distance = 0.0
for i in range(len(UTM_path)):
    delta_x = UTM_path[i][0] - UTM_path[(i+1)%len(UTM_path)][0]
    delta_y = UTM_path[i][1] - UTM_path[(i+1)%len(UTM_path)][1]
    distance_to_next_waypoint = np.sqrt(delta_x*delta_x + delta_y*delta_y)
    if distance_to_next_waypoint == 0.0:
       print(i, "entry has length == 0.0 to last waypoint")
    UTM_path[i].append(distance_to_next_waypoint)
    UTM_path[i].append(total_lap_distance)
    total_lap_distance += distance_to_next_waypoint

diffx = 0 # -989.70
diffy = 0 # -58.984


pi = 3.14159265358979
def RadToDeg (rad):
    return (rad / pi * 180.0)

def toGPS(simx, simy):
    projx, projy = simx+diffx, simy + diffy
    lon, lat = mt(projx, projy,inverse=True)
    lon += -0.006893478893516658
    lat += -0.0013556108003527356
    return [lon, lat]

# sim to UTM offset in meters
offsetx = 555578.648369298
offsety = 4375769.19238589
def toUTM(simx, simy):
    simx = 1241.2 + (1241.2 - simx)*1.01
    simy = 860.9 + (860.9 - simy)*1.05
    return [simx+offsetx, simy+offsety]

def UTMtoLatLon(utmx, utmy):
    lat, lon = utm.to_latlon(utmx, utmy, 10, 'T')
    return lon, lat

def LatLontoUTM(lon, lat):
    utmx, utmy, zone, zoneletter = utm.from_latlon(lat, lon)
    return [utmx, utmy]


# CTE and distance calculations
def sqr(x):
    return x * x

def dist2(v, w):
    return sqr(v[0] - w[0]) + sqr(v[1] - w[1])

def midPointSegment(p, v, w):
    l2 = dist2(v, w)
    if (l2 == 0):
        return v

    t = ((p[0] - v[0]) * (w[0] - v[0]) + (p[1] - v[1])*(w[1] - v[1])) / l2
    t = np.max([0, np.min([1, t])])
    midp = [(v[0]+t*(w[0]-v[0])), (v[1]+t*(w[1]-v[1]))]
    return midp

def distToSegmentSquared(p, v, w):
    midp = midPointSegment(p, v, w)
    return dist2(p, midp)

def distToSegment(p, v, w):
    return np.sqrt(distToSegmentSquared(p, v, w))

def closestPathSegmentMidpoint(nk, p):
    kp0 = UTM_path[nk%len(UTM_path)]
    kp1 = UTM_path[(nk+1)%len(UTM_path)]
    return midPointSegment(p, kp0, kp1)

def distToPathSegment(nk, p):
    kp0 = UTM_path[nk%len(UTM_path)]
    kp1 = UTM_path[(nk+1)%len(UTM_path)]
    return distToSegment(p, kp0, kp1)

def sideOfPoint(nk, p):
    kp0 = UTM_path[nk%len(UTM_path)]
    kp1 = UTM_path[(nk+1)%len(UTM_path)]
    d = (p[0]-kp0[0])*(kp1[1]-kp0[1])-(p[1]-kp0[1])*(kp1[0]-kp0[0])
    if d < 0.0:
        return -1
    return 1

# find nearest waypoint and its CTE and distance to next waypoint
def NearestWayPointCTEandDistance(p):
    distance_to_waypoint = 1000.0

    k=0
    for i in range(len(UTM_path)):
        distance = distToPathSegment(i, p)
        if distance < distance_to_waypoint:
            distance_to_waypoint = distance
            k=i

    # get closest midpoint
    midp = closestPathSegmentMidpoint(k, p)

    # calculate CTE and distance to next waypoint
    distance_to_next_waypoint = np.sqrt(dist2(UTM_path[(k+1)%len(UTM_path)], p))
    CTE = sideOfPoint(k, p)*distToPathSegment(k, p)

    # calculate current lap distance
    lap_distance = UTM_path[k][3]
    lap_distance += np.sqrt(dist2(UTM_path[k], midp))

    # return results
    return CTE, distance_to_next_waypoint, lap_distance



def generate_thunderhill_batches(gen, batch_size):
    batch_x = []
    batch_x2 = []
    batch_x3 = []
    batch_y = []
    while True:
        for img, steering_angle, throttle, brake, speed, longitude, latitude in gen:
            for img1, steering_angle1 in RandomFlip(img, steering_angle):
                img1, steering_angle1 = RandomBrightness(img1, steering_angle1)
                img1 = Preproc(img1)
                img1, steering_angle1, throttle1 = RandomShift(img1, steering_angle1,throttle)


                brake1=brake

                if steering_angle1>0.5:
                    brake1=1
                    throttle1=0
                if steering_angle1<-0.5:
                    brake1=1
                    throttle1=0

                if np.random.uniform() < 0.5:
                    speed1 = speed * np.random.uniform()
                else:
                    speed1 = speed

                speed_cl=np.array(speedToClass(speed))
                speed_cl1=np.array(speedToClass(speed1))

                if speed1 < 20*0.44704:
                    throttle1=1
                if speed1 > 50*0.44704:
                    throttle1=0
                if speed1 > 70*0.44704:
                    throttle1=0
                    brake=1

                if brake>0.7:
                    throttle=0

                # if speed_cl[-2]==1 and steering_angle1!=0:
                #     throttle1=0
                # if speed_cl[-1]==1 and steering_angle1!=0:
                #     brake1=1
                #     throttle1=0

                batch_x.append(np.reshape(img1, (1, HEIGHT, WIDTH, DEPTH)))
                batch_x2.append(speed_cl1)


                UTMp = LatLontoUTM(RadToDeg(longitude), RadToDeg(latitude))
                CTE, Distance2NextWaypoint, LapDistance = NearestWayPointCTEandDistance(UTMp)

                CTE/=5.0
                LapDistance=2*(LapDistance/total_lap_distance)-1

                font = cv2.FONT_HERSHEY_SIMPLEX
                if show_images:
                    imgp = Preproc(img.copy()[::-1,:,:])
                    imgshow=(np.concatenate((imgp,img1[::-1,:,:]),axis=0)+0.5)*0.5

                    imgshow = cv2.resize(imgshow,(0,0),fx=2,fy=2)
                    cv2.putText(imgshow,'%s'%(str(list(speed_cl))),(10,30), font, 0.8,(0,0,255),2,cv2.LINE_AA)
                    cv2.putText(imgshow,'%s'%(str(list(speed_cl1))),(10,190), font, 0.8,(0,0,255),2,cv2.LINE_AA)
                    cv2.putText(imgshow,'%.3f %.1f %.1f %.1f'%(steering_angle,throttle, brake,speed),(10,140), font, 0.8,(0,0,255),2,cv2.LINE_AA)
                    cv2.putText(imgshow,'%.3f %.1f %.1f %.1f'%(steering_angle1,throttle1, brake1,speed1),(10,300), font, 0.8,(0,0,255),2,cv2.LINE_AA)

                    cv2.putText(imgshow,'%.5f %.5f'%(longitude, latitude),(10,110), font, 0.8,(0,0,255),2,cv2.LINE_AA)
                    cv2.putText(imgshow,'%.3f %.3f'%(CTE,LapDistance),(10,250), font, 0.8,(0,0,255),2,cv2.LINE_AA)
                    cv2.imshow('img',imgshow)
                    cv2.waitKey(0)

                batch_x3.append((CTE, LapDistance))

                batch_y.append([steering_angle1,throttle1, brake1])
                if len(batch_x) == batch_size:
                    batch_x, batch_x2, batch_x3, batch_y = shuffle(batch_x, batch_x2, batch_x3, batch_y)
                    yield [np.vstack(batch_x), np.vstack(batch_x2), np.vstack(batch_x3)], np.vstack(batch_y)
                    batch_x = []
                    batch_x2 = []
                    batch_x3 = []
                    batch_y = []
def getSession5(repo):
    csvfname=repo+'/dataset_session_5/output.csv'
    data=None
    with open(csvfname, 'r') as csvfile:
        data = list(csv.reader(csvfile, delimiter=','))
        dd='/'.join(csvfname.split('/')[:-1])
        data = [(dd+'/'+x[0],float(x[3]),float(x[4]),float(x[5]),float(x[6])) for x in data]
    return data

sim320csvs=['dataset_sim_000_km_few_laps/driving_log.csv','dataset_sim_001_km_320x160/driving_log.csv','dataset_sim_002_km_320x160_recovery/driving_log.csv','dataset_sim_003_km_320x160/driving_log.csv','dataset_sim_004_km_320x160_cones_brakes/driving_log.csv']

polysynccsvs=['dataset_polysync_1464466368552019/output.txt','dataset_polysync_1464470620356308/output.txt','dataset_polysync_1464552951979919/output.txt']

def getSim320(repo,rel_csv):
    csvfname=repo+rel_csv
    data=[]
    with open(csvfname, 'r') as csvfile:
        data_tmp = list(csv.reader(csvfile, delimiter=','))
        for row in data_tmp:
            x7=[float(x) for x in row[7].split(':')]
            x8=[float(x) for x in row[8].split(':')]
            dd='/'.join(csvfname.split('/')[:-1])
            data.append((dd+'/'+row[0],float(row[3]),float(row[4]),float(row[5]),float(row[6])))
    return data

def getDataFromFolder(folder):
    data5=getSession5(folder)
    data000=getSim320(folder,sim320csvs[0])
    data001=getSim320(folder,sim320csvs[1])
    data002=getSim320(folder,sim320csvs[2])
    print('Dataset Session5 size:',len(data5))
    print('Dataset 000 size:',len(data000))
    print('Dataset 001 size:',len(data001))
    print('Dataset 002 size:',len(data002))
    # data=data5 + data001 + data002
    # data=data5[::5]
    # df = shuffle(data)
    # return train_test_split(df, test_size=0.2, random_state=42)
    return shuffle(data001),shuffle(data002)
    # return shuffle(data5[::10]),shuffle(data001)

def genSession5(folder):
    data=shuffle(getSession5(folder))
    while True:
        for row in data:
            img = cv2.imread(row[0])[:200,200:-200,:]
            steering_angle = row[1]
            yield img,steering_angle/2.0,row[2],row[3],row[4]

def genSim001(folder):
    data=shuffle(getSim320(folder,sim320csvs[1]))
    while True:
        for row in data:
            img = cv2.imread(row[0])[20:140,:,:]
            steering_angle = row[1]
            yield img,steering_angle,row[2],row[3],row[4]

def genSim002(folder):
    data=shuffle(getSim320(folder,sim320csvs[2]))
    while True:
        for row in data:
            img = cv2.imread(row[0])[20:140,:,:]
            steering_angle = row[1]
            yield img,steering_angle,row[2],row[3],row[4]

def genSim003(folder):
    data=shuffle(getSim320(folder,sim320csvs[2]))
    while True:
        for row in data:
            img = cv2.imread(row[0])[20:140,:,:]
            steering_angle = row[1]
            yield img,steering_angle,row[2],row[3],row[4]

def getPolysync(repo, rel_csv):
    csvfname=repo+rel_csv
    data=[]
    with open(csvfname, 'r') as csvfile:
        data_tmp = list(csv.reader(csvfile, delimiter=','))[1:]
        dd='/'.join(csvfname.split('/')[:-1])
        for row in data_tmp:
            data.append((dd+'/'+row[0],float(row[-3]),float(row[-2]),float(row[-1]),float(row[-6])))
    return data

def genPolysync0(folder):
    data=shuffle(getPolysync(folder,polysynccsvs[0])[500:-500])
    while True:
        for row in data:
            img = cv2.imread(row[0])[-250:1:-1,300:-300,:]
            steering_angle = row[1]
            yield img,steering_angle/5.0,0.8,row[3],row[4]

def genPolysync2(folder):
    data=getPolysync(folder,polysynccsvs[2])
    data=shuffle(data[500:2000] + data[3100:-500])
    while True:
        for row in data:
            img = cv2.imread(row[0])[-250:1:-1,300:-300,:]
            steering_angle = row[1]
            yield img,steering_angle/5.0,0.8,row[3],row[4]

def genAll(folder):
    datasets=[]
    # datasets.append(shuffle(getPolysync(folder,polysynccsvs[0])[500:-500]))
    # data=getPolysync(folder,polysynccsvs[2])
    # data=shuffle(data[500:2000] + data[3100:-500])
    # datasets.append(data)
    datasets.append(shuffle(getSim320(folder,sim320csvs[2])))
    datasets.append(shuffle(getSim320(folder,sim320csvs[3])))
    datasets.append(shuffle(getSim320(folder,sim320csvs[4])))
    datasets.append(shuffle(getSession5(folder)))

    while True:
        nr=0
        ex=int(np.random.uniform() * len(datasets[nr]))
        row=datasets[nr][ex]
        img = cv2.imread(row[0])[20:140,:,:]
        steering_angle = row[1]
        yield img,steering_angle,row[2],row[3],row[4]
        nr=nr+1
        ex=int(np.random.uniform() * len(datasets[nr]))
        row=datasets[nr][ex]
        img = cv2.imread(row[0])[20:140,:,:]
        steering_angle = row[1]
        yield img,steering_angle,row[2],row[3],row[4]
        nr=nr+1
        ex=int(np.random.uniform() * len(datasets[nr]))
        row=datasets[nr][ex]
        img = cv2.imread(row[0])[20:140,:,:]
        steering_angle = row[1]
        yield img,steering_angle,row[2],row[3],row[4]
        nr=nr+1
        ex=int(np.random.uniform() * len(datasets[nr]))
        row=datasets[nr][ex]
        img = cv2.imread(row[0])[:200,200:-200,:]
        steering_angle = row[1]
        yield img,steering_angle/2.0,row[2],row[3],row[4]

thunderhill_data_dir_1='/home/karol/projects/udacity/Racing/thunderhill-day1-data/*'
thunderhill_datasets_1=glob(thunderhill_data_dir_1)
thunderhill_data_dir_2='/home/karol/projects/udacity/Racing/thunderhill-day2-data/*'
thunderhill_datasets_2=glob(thunderhill_data_dir_2)
# ,path,heading,longitude,latitude,quarternion0,quarternion1,quarternion2,quarternion3,vel0,vel1,vel2,acc0,acc1,acc2,steering,throttle,brake,speed,accel
def genThDay1():
    while True:
        for dataset in thunderhill_datasets_1:
            with open(dataset + '/output_processed.txt', 'r') as csvfile:
                data_tmp = shuffle(list(csv.reader(csvfile, delimiter=','))[1:])
                dd='/'.join(dataset.split('/'))
                for row in data_tmp:
                    if row[-1] == '':
                        continue
                    # row=row[1:]
                    img = cv2.imread(dataset + '/' + row[0]) # image
                    steering_angle = float(row[14])
                    throttle = float(row[-1])
                    brake = float(row[16])
                    speed = float(row[17])
                    longtitude = float(row[2])
                    latitude = float(row[3])

                    if throttle<-0.1:
                        throttle=-0.1
                    throttle=throttle+0.1
                    throttle=throttle/0.3
                    if throttle>1:
                        throttle=1

                    yield img,steering_angle, throttle, brake, speed, longtitude, latitude
                    break
def genThDay2():
    while True:
        for dataset in thunderhill_datasets_2:
            with open(dataset + '/output_processed.txt', 'r') as csvfile:
                data_tmp = shuffle(list(csv.reader(csvfile, delimiter=','))[1:])
                dd='/'.join(dataset.split('/'))
                for row in data_tmp:
                    if row[-1] == '':
                        continue
                    # row=row[1:]
                    img = cv2.imread(dataset + '/' + row[0]) # image
                    steering_angle = float(row[14])
                    throttle = float(row[-1])
                    brake = float(row[16])
                    speed = float(row[17])
                    longtitude = float(row[2])
                    latitude = float(row[3])

                    if throttle<-0.1:
                        throttle=-0.1
                    throttle=throttle+0.1
                    throttle=throttle/0.3
                    if throttle>1:
                        throttle=1

                    yield img,steering_angle, throttle, brake, speed, longtitude, latitude
                    break
