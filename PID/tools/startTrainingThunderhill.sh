#!/bin/sh
WID=$(/usr/bin/xdotool search --name "self_driving_car_nanodegree_program" | head -n 1)
/usr/bin/xdotool mousemove --window $WID --sync 300 300
/usr/bin/xdotool click 1
sleep 1
/usr/bin/xdotool mousemove --window $WID --sync 300 500
/usr/bin/xdotool click 1

