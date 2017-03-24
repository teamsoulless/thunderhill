# C++ PID Controller

C++ PID Controller with Realtime Twittle Trainer for Udacity Simulator for Ubuntu.  May work in other OS, but have not tried it.

## Dependencies

This original C++ project came from [https://github.com/udacity/CarND-PID-Control-Project](https://github.com/udacity/CarND-PID-Control-Project).  Please make sure you have all dependencies installed.  Make sure you have `libuv1-dev` installed before attempting to build uWebSockets-0.13 (uWS).

```
$ sudo apt-get install libuv1-dev
```

And update the libuWS.so location if it gets installed in `/usr/lib64/libuWS.so`

```
sudo ln -s /usr/lib64/libuWS.so /usr/lib/libuWS.so
```

Install `/usr/bin/xdotool` which is needed to control the starting and stoping of the simulator sessions.

```
$ sudo apt-get install xdotool
```

**TODOs:**
1. Modify Simulator or src/main.cpp to read or calculate the CTE needed to compute the total error for Twittle parameter calculation for Thunderhill track.
2. Modify src/main.cpp to adjust trottle other than cruise control at ~40 MPH.

## Basic Build Instructions

```
$ mkdir build
$ cd build/
$ cmake ..
$ make
```

## Basic Training Instruction

1. Start Udacity Simulator for Thunderhill
```
$ ./UdacitySRCTeamSimLinux.x86_64
```
2. Set it to lowest resolution.  (This is needed for start and stop script to press the right buttons for session automation)
![Low Resolution Selection](./images/lowestres.png)
![Thunderhill Simulator](./images/thunderhill-simulator.png)
3. Start the Twittle session.
```
$ cd tools
$ python twittle.py
```
[![Twittle Video](./images/twittlevideo.png)](https://youtu.be/oBRk1Nj6Cao)