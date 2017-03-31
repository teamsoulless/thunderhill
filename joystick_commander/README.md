### joystick_commander

This example enables a joystick to control a vehicle, and provides an example of a high level control node in PolySync. 

If you want to create a high level control node to command a vehicle, this is the best place to start.  

This example sends `ps_platform_brake_command_msg`,  `ps_platform_steering_command_msg`, and 
`ps_platform_throttle_command_msg` out on the PS bus at 50Hz.

The actuation commands are calculated from the joystick and trigger button positions for steering, brake, and throttle control.
  
It can also do high level management of control states; “enables” and "disables" vehicle control.

### Dependencies

Packages: libglib2.0-dev libsdl2-dev

To install on Ubuntu: 

```bash
sudo apt-get install <package>
```

### Building and running the node

```bash
$ cd joystick_commander
$ make
$ ./bin/polysync-joystick-commander 
```

For more API examples, visit the "Tutorials" and "Development" sections in the PolySync Help Center [here](https://help.polysync.io/articles/).
