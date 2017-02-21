#!/usr/bin/python3
from collections import deque
import matplotlib.pyplot as plt
import numpy as np
import cv2
import sys
import json

throttle_window=(277,520,690,691)
steering_left_window=(540,654,686,687)
steering_right_window=(656,735,686,687)
brakes_window=(770,970,686,687)

def main():
    fname = sys.argv[1]
    cap = cv2.VideoCapture(fname)
    # Define the codec and create VideoWriter object
    # fourcc = cv2.VideoWriter_fourcc(*'X264')
    # video_out = cv2.VideoWriter('results/'+fname.split('/')[-1],fourcc, 25, (1280,720))

    if cap.isOpened():
        ret, img = cap.read()
        if not img is None:
            cv2.imshow('img',img)
            cv2.waitKey(1)

    while(cap.isOpened()):
        ret, img = cap.read()
        if not img is None:
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

            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(img,'%.0f'%(100*throttle_value),(throttle_window[0]+100,718), font, 1,(0,0,255),2,cv2.LINE_AA)
            cv2.putText(img,'%.0f'%(100*steering),(steering_right_window[0],718), font, 1,(0,0,255),2,cv2.LINE_AA)
            cv2.putText(img,'%.0f'%(100*brakes),(brakes_window[0],718), font, 1,(0,0,255),2,cv2.LINE_AA)
            cv2.imshow('img',img)
            cv2.waitKey(50)
        # write result
        # video_out.write(res)
    cap.release()
    # video_out.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    if len(sys.argv)==2:
        main()
    else:
        print('\nOne argument required: path_to_Session_5_-_Thunderhill_West.mp4\nExit\n')
