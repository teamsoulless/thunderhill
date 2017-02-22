#!/usr/bin/python3
import matplotlib.pyplot as plt
from operator import itemgetter
from tqdm import tqdm
import numpy as np
import cv2
import os
import sys
import csv
import json

#format: (xmin,xmax,ymin,ymax)
throttle_window=(277,520,690,691)
steering_left_window=(540,654,686,687)
steering_right_window=(656,735,686,687)
brakes_window=(770,970,686,687)
save_window=(0,1280,280,600)

left_digit=(17,45,660,700)
right_digit=(46,78,660,700)

#csv arrays - zip to get rows
throttle_csv=[]
steering_csv=[]
brakes_csv=[]
speed_csv=[]
filenames_csv=[]

#show data?
show=False

font_recog = cv2.FONT_HERSHEY_PLAIN
font = cv2.FONT_HERSHEY_SIMPLEX

#output images dir
img_dir='images'
try:
    os.mkdir(img_dir)
except IOError as e:
    if e.errno!=17:
        print(e)
        raise
    else:
        print('Directory "%s"'%img_dir,'already exists! Will overwrite files!')

def dataGen(cap):
    while(cap.isOpened()):
        ret, img = cap.read()
        if not img is None:
            img_raw=img.copy()
            thr=img[throttle_window[2]:throttle_window[3],throttle_window[0]:throttle_window[1],0]
            thr[thr<=50]=0
            thr[thr>50]=255
            throttle_value=np.sum(thr)/(255.0 * (throttle_window[1]-throttle_window[0]))

            st_l=img[steering_left_window[2]:steering_left_window[3],steering_left_window[0]:steering_left_window[1],0]
            st_l[st_l<=80]=0
            st_l[st_l>80]=255
            st_lv=1.95*np.sum(st_l)/(255.0 * (steering_left_window[1]-steering_left_window[0]))

            st_r=img[steering_right_window[2]:steering_right_window[3],steering_right_window[0]:steering_right_window[1],0]
            st_r[st_r<=80]=0
            st_r[st_r>80]=255
            st_rv=1.38 * np.sum(st_r)/(255.0 * (steering_right_window[1]-steering_right_window[0]))

            if st_rv>st_lv:
                steering=st_rv
            else:
                steering=-st_lv

            if np.abs(steering)<0.01:
                steering=0

            br=img[brakes_window[2]:brakes_window[3],brakes_window[0]:brakes_window[1],0]
            br[br<=80]=0
            br[br>80]=255
            brakes=1.09*np.sum(br)/(255.0 * (brakes_window[1]-brakes_window[0]))

            font_weight=2

            dgt_l=img[left_digit[2]:left_digit[3],left_digit[0]:left_digit[1],:]
            dgt_l[dgt_l[:,:,0]<=200]=0
            dgt_l[dgt_l[:,:,0]>200]=255
            digits=[1000]*10
            for d in range(10):
                copy=dgt_l.copy()
                render=np.zeros_like(copy)
                cv2.putText(render,str(d),(-2,render.shape[0]-3), font_recog, 3.3,(255,255,255),font_weight,cv2.LINE_AA)
                mask_and= cv2.bitwise_and(copy,render)
                mask_sum= copy.copy()
                mask_sum[render>100]=255
                digits[d]= np.count_nonzero(mask_and[:,:,0]) / np.count_nonzero(mask_sum[:,:,0])


            ld=max(enumerate(digits), key=itemgetter(1))[0]
            # render=np.zeros_like(render)
            # cv2.putText(render,str(ld),(-2,render.shape[0]-3), font_recog, 3.3,(255,255,255),font_weight,cv2.LINE_AA)
            # cv2.imshow('dgt_l',np.concatenate((dgt_l,render),axis=1))
            # cv2.waitKey(1)


            dgt_r=img.copy()[right_digit[2]:right_digit[3],right_digit[0]:right_digit[1],:]
            dgt_r[dgt_r[:,:,0]<=200]=0
            dgt_r[dgt_r[:,:,0]>200]=255
            digits=[1000]*10
            for d in range(10):
                copy=dgt_r.copy()
                render=np.zeros_like(copy)
                cv2.putText(render,str(d),(-1,render.shape[0]-4), font_recog, 3.25,(255,255,255),font_weight,cv2.LINE_AA)
                mask_and= cv2.bitwise_and(copy,render)
                mask_sum= copy.copy()
                mask_sum[render>100]=255
                digits[d]=np.count_nonzero(mask_and[:,:,0]) / np.count_nonzero(mask_sum[:,:,0])
                # cv2.imshow('dgt_r',np.concatenate((dgt_r,render,mask_and,mask_sum),axis=1))
                # cv2.waitKey(100)


            rd=max(enumerate(digits), key=itemgetter(1))[0]

            speed=ld*10+rd

            # render=np.zeros_like(render)
            # render=np.concatenate((render,render),axis=1)
            # cv2.putText(render,str(speed),(-1,render.shape[0]-4), font_recog, 3.25,(0,0,255),font_weight,cv2.LINE_AA)
            # cv2.imshow('dgt_r',np.concatenate((dgt_l,dgt_r,render),axis=1))
            # cv2.waitKey(1)

            #Save img
            fname=img_dir+'/center_%08d.jpg'%len(filenames_csv)
            to_save=img[save_window[2]:save_window[3],save_window[0]:save_window[1],:]
            cv2.imwrite(fname,to_save)

            throttle_csv.append(throttle_value)
            steering_csv.append(steering)
            brakes_csv.append(brakes)
            speed_csv.append(speed)
            filenames_csv.append(fname)

            if show:
                cv2.putText(img_raw,'%.0f'%(100*throttle_value),(throttle_window[0]+100,718), font, 1,(0,0,255),2,cv2.LINE_AA)
                cv2.putText(img_raw,'%.0f'%(100*steering),(steering_left_window[0],718), font, 1,(0,0,255),2,cv2.LINE_AA)
                cv2.putText(img_raw,'%.0f'%(100*brakes),(brakes_window[0],718), font, 1,(0,0,255),2,cv2.LINE_AA)
                cv2.putText(img_raw,str(speed),(25,718), font, 1,(0,0,255),2,cv2.LINE_AA)
                cv2.imshow('img',img_raw)
                cv2.waitKey(1)

            row=[fname,'','',steering,throttle_value,brakes,speed]
            yield row

def main():
    fname = sys.argv[1]
    cap = cv2.VideoCapture(fname)
    length=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    with open('output.csv', "w") as f:
        writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_ALL)
        counter=0
        total=length-20
        for row in tqdm(dataGen(cap),total=total): #without last 20 frames...
            writer.writerow(row)
            counter=counter+1
            if counter>total:
                break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    if len(sys.argv)==2:
        main()
    else:
        print('\nOne argument required: path_to_Session_5_-_Thunderhill_West.mp4\nExit\n')
