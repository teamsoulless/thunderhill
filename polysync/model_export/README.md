## Loading Tensorflow graph in C++

This code is extended from ["Loading a TensorFlow graph with the C++ API"](https://medium.com/jim-fleming/loading-a-tensorflow-graph-with-the-c-api-4caaff88463f#.6dgk6x8d6)

Requirements:

1. [Install Bazel](https://bazel.build/versions/master/docs/install.html)
2. [Clone Tensorflow repo](https://github.com/tensorflow/tensorflow): <br/>
`git clone --recursive https://github.com/tensorflow/tensorflow`
3. Create sample graph using _xor.py_
4. Copy _loader_ folder into tensorflow root folder (from step 2) or symbolically link it.
5. Compile and run:
  - from the root of the tensorflow repo, run `./configure`
  - from loader folder call `bazel build :loader`. If this results in an error try `bazel run //tensorflow/cc/loader:loader` from the repository root. 
  - from the repository root, go into `bazel-bin/tensorflow/loader`
  - copy the graph protobuf from step 3 to _loader/models/graph.pb_
  - run `./loader` and check the output
