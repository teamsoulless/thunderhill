#include "PID.h"

using namespace std;

/*
* PID class.
*/

PID::PID() {}

PID::~PID() {}

/*
* Initialize PID.
*/
void PID::Init(double Kp, double Ki, double Kd) {
  PID::Kp = Kp;
  PID::Ki = Ki;
  PID::Kd = Kd;

  p_error = -10000.0;
  i_error = 0.0;
  d_error = 0.0;
}

/*
* Update the PID error variables given cross track error.
*/
void PID::UpdateError(double cte) {
  if (p_error == -10000.0) {
    p_error = cte;
  }
  d_error = cte - p_error;
  p_error = cte;
  i_error += cte;
}

/*
* Calculate the total PID error.
*/
double PID::TotalError() {
  return -Kp*p_error - Kd*d_error - Ki*i_error;
}

