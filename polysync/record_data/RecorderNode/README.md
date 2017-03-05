### RecorderNode

This node subscribes to ps_platform_motion_msg and ps_image_data_msg and writes the data to _recording.txt) file located in the build directory. Resulting _recording.txt_ file is large (approx 1GB for 1 minute of data replay)

### Dependencies

Packages: libglib2.0-dev

To install on Ubuntu:

```bash
sudo apt-get install <package>
```

### Building and running the node

```bash
$ cd RecorderNode
$ mkdir build && cd build
$ cmake ..
$ make
$ ./recorder-node
```

For more API examples, visit the "Tutorials" and "Development" sections in the PolySync Help Center [here](https://help.polysync.io/articles/).
