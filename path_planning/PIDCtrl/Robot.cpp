//
// Created by Maruf Maniruzzaman on 3/13/17.
// Adapted From AI for Robotics course:
// https://classroom.udacity.com/courses/cs373/lessons/48743150/concepts/487283460923#
//

#include <math.h>
#include <cstdlib>
#include "Robot.h"
#include <random>
#include <functional>

std::random_device rd;
std::mt19937 e2(rd());

float gauss(float mu, float sigma) {
    std::normal_distribution<float> dist(mu, sigma);
    auto gen_gaussian = std::bind(dist, e2);
    float number = gen_gaussian();

    return number;
}

Robot::Robot():Robot(20.0)
{
}

/*
creates robot and initializes location/orientation to 0, 0, 0
*/
Robot::Robot(float length) {
    this->x = 0.0;
    this->y = 0.0;
    this->orientation = 0.0;
    this->length = length;
    this->steering_noise = 0.0;
    this->distance_noise = 0.0;
    this->steering_drift = 0.0;
}

/*
sets a robot coordinate
*/
void Robot::set(float new_x, float new_y, double new_orientation) {
    this->x = float(new_x);
    this->y = float(new_y);
    this->orientation = fmod(new_orientation, (2.0 * pi));
}

/*
sets the noise parameters
*/
void Robot::set_noise(float new_s_noise, float new_d_noise) {
    // makes it possible to change the noise parameters
    // this is often useful in particle filters
    this->steering_noise = float(new_s_noise);
    this->distance_noise = float(new_d_noise);
}


/*
# --------
# set_steering_drift:
#	sets the systematical steering drift parameter
#
*/

void Robot::set_steering_drift(float drift) {
    this->steering_drift = drift;
}

/*
#    steering = front wheel steering angle, limited by max_steering_angle
#    distance = total distance driven, most be non-negative
*/
Robot Robot::move(float steering, float distance, float tolerance, float max_steering_angle) {

    if (steering > max_steering_angle) {
        steering = max_steering_angle;
    }
    if (steering < -max_steering_angle) {
        steering = -max_steering_angle;
    }

    if (distance < 0.0) {
        distance = 0.0;
    }

    // make a new copy
    Robot res = Robot();
    res.length = this->length;
    res.steering_noise = this->steering_noise;
    res.distance_noise = this->distance_noise;
    res.steering_drift = this->steering_drift;

    // apply noise
    float steering2 = gauss(steering, this->steering_noise);
    float distance2 = gauss(distance, this->distance_noise);

    // apply steering drift
    steering2 += this->steering_drift;

    // Execute motion
    float turn = tan(steering2) * distance2 / res.length;

    if (abs(turn) < tolerance) {
        //approximate by straight line motion

        res.x = this->x + (distance2 * cos(this->orientation));
        res.y = this->y + (distance2 * sin(this->orientation));
        res.orientation = fmod((this->orientation + turn), (2.0 * pi));
    } else {
        // approximate bicycle model for motion
        float radius = distance2 / turn;
        float cx = this->x - (sin(this->orientation) * radius);
        float cy = this->y + (cos(this->orientation) * radius);
        res.orientation = fmod((this->orientation + turn), (2.0 * pi));
        res.x = cx + (sin(res.orientation) * radius);
        res.y = cy - (cos(res.orientation) * radius);
        return res;
    }
}

std::string Robot::ToString() {
    char buff[80];
    snprintf(buff, sizeof(buff),"[x=%.5f y=%.5f orient=%.5f]", this->x, this->y, this->orientation);
    std::string strRep = buff;
    return strRep;
}
