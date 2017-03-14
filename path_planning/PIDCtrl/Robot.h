//
// Created by Maruf Maniruzzaman on 3/13/17.
// Adapted From AI for Robotics course:
// https://classroom.udacity.com/courses/cs373/lessons/48743150/concepts/487283460923#
//

#ifndef PID_ROBOT_H
#define PID_ROBOT_H
#include<string>

#define pi  3.14159
#define piover4  pi / 4.0

float gauss(float mu, float sigma);

class Robot {
public:
    Robot();
    Robot(float length);
    void set(float new_x, float new_y, float new_orientation);
    void set_noise(float new_s_noise, float new_d_noise);
    void set_steering_drift(float drift);
    Robot move(float steering, float distance, float tolerance = 0.001, float max_steering_angle = piover4);
    std::string ToString();
    void set(float new_x, float new_y, double new_orientation);

    double orientation;
    float y;
    float x;
    float steering_drift;
    float distance_noise;
    float steering_noise;
    float length;
};


#endif //PID_ROBOT_H
