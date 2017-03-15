//
// Created by Maruf Maniruzzaman on 3/13/17.
// Adapted From AI for Robotics course:
// https://classroom.udacity.com/courses/cs373/lessons/48743150/concepts/487283460923#
//

#include <iostream>
#include "Robot.h"

void PID(float param1, float param2, float param3) {
    Robot myrobot = Robot();
    myrobot.set(0.0, 1.0, 0.0);
    float speed = 1.0;
    int N = 100;
    myrobot.set_steering_drift(10.0 / 180.0 * pi);
    float int_cte = 0.0;
    float cte = myrobot.y;

    for(int i=0; i<N; i++) {
        float dcte = myrobot.y - cte;
        cte = myrobot.y;
        int_cte += cte;
        float steer = -param1 * cte - param2 * dcte - param3 * int_cte;
        myrobot = myrobot.move(steer, speed);

        std::cout << myrobot.ToString() << steer << std::endl;
    }
}

int main() {
    PID(0.2, 3.0, 0.01);
    return 0;
}