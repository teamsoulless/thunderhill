#!/bin/sh
WID=$(/usr/bin/xdotool search --name "self_driving_car_nanodegree_program" | head -n 1)
/usr/bin/xdotool key --window $WID Escape

