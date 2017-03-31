This is a smaller version of controller.
In C++ it creates a shared lib, which listens to image_data message.
In Python it passes the message to the trained model and makes predictions.
TODO:
1. replace currently used model with model which predicts not only steering angle, but also break and throttle (the model can be very simple)
2. issue steering, break and throttle commands (see example for steering in MainNode) from the MainNode.cpp (or can only be steering command, most important is to issue any command)

To run:
1. in MainNode folder:<br/>
cmake .<br/>
make

2. in SmallController folder:<br/>
python clientnode.py

3. start session in PS studio
