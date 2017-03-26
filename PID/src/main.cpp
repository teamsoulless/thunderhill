#include <uWS/uWS.h>
#include <iostream>
#include "json.hpp"
#include "PID.h"
#include <math.h>

using namespace std;

// for checking augments for enabling TWITTLE
bool check_arguments(int argc, char* argv[]) {
  string usage_instructions = "Usage instructions: ";
  usage_instructions += argv[0];
  usage_instructions += "[kp ki kd]\n";
  usage_instructions += "  kp: optional \n";
  usage_instructions += "  ki: optional process lidar only flag\n";
  usage_instructions += "  kd: optional print final P_ flag\n";
  usage_instructions += "kp ki and kd must be given together or not at all\n";
  bool has_valid_args = false;
  bool has_pid_args = false;

  // make sure the user has provided input and output files
  if (argc == 1) {
    has_valid_args = true;
  } else if (argc < 4) {
    cerr << "Please include all PID parameters.\n" << usage_instructions << endl;
  } else if (argc == 4) {
    has_valid_args = true;
    has_pid_args = true;
  } else if (argc > 4) {
    cerr << "Too many arguments.\n" << usage_instructions << endl;
  }

  if (!has_valid_args) {
    exit(EXIT_FAILURE);
  }

  return has_pid_args;
}

// for convenience
using json = nlohmann::json;

// For converting back and forth between radians and degrees.
constexpr double pi() { return atan(1)*4; }
double deg2rad(double x) { return x * pi() / 180; }
double rad2deg(double x) { return x * 180 / pi(); }

// Checks if the SocketIO event has JSON data.
// If there is data the JSON object in string format will be returned,
// else the empty string "" will be returned.
string hasData(string s) {
  auto found_null = s.find("null");
  auto b1 = s.find_first_of("[");
  auto b2 = s.find_last_of("]");
  if (found_null != string::npos) {
    return "";
  }
  else if (b1 != string::npos && b2 != string::npos) {
    return s.substr(b1, b2 - b1 + 1);
  }
  return "";
}

int main(int argc, char* argv[])
{
  uWS::Hub h;
  bool enableTwittle = false;
  long ncount = 0;
  long ccount = 0;
  double twittle_err = 0;

  double kp = 0.21;
  double ki = 0.005;
  double kd = 3.05;

  if (check_arguments(argc, argv)) {
    kp = stod(argv[1]);
    ki = stod(argv[2]);
    kd = stod(argv[3]);
    enableTwittle = true;

    // number of samples per lap around the lake track (to bridge).
    ncount = 2125;
    ccount = 0;
  }

  PID pidsteer;
  PID pidthrottle;
  // Initialize the pid variable.
  // PID for steering
  pidsteer.Init(kp, ki, kd);

  // PID for throttle - could be changed if we have a speed CTE too.
  pidthrottle.Init(0.2, 0.0, 1.0);

  // currently holding it constant.
  double desired_speed = 40;

  h.onMessage([&pidsteer, &pidthrottle, &desired_speed, &enableTwittle, &ncount, &ccount, &twittle_err](uWS::WebSocket<uWS::SERVER> ws, char *data, size_t length, uWS::OpCode opCode) {
    // "42" at the start of the message means there's a websocket message event.
    // The 4 signifies a websocket message
    // The 2 signifies a websocket event
    if (length && length > 2 && data[0] == '4' && data[1] == '2')
    {
      auto s = hasData(string(data));
      if (s != "") {
        auto j = json::parse(s);
        string event = j[0].get<string>();
        if (event == "telemetry") {
          // j[1] is the data JSON object
          double cte = -stod(j[1]["cte"].get<string>());
          double speed = stod(j[1]["speed"].get<string>());
          double angle = stod(j[1]["steering_angle"].get<string>());
          double steer_value;
          /*
          * Calcuate steering value here, remember the steering value is
          * [-1, 1].
          * NOTE: Feel free to play around with the throttle and speed. Maybe use
          * another PID controller to control the speed!
          */
          pidsteer.UpdateError(cte);
          pidthrottle.UpdateError(speed-desired_speed);
          steer_value = pidsteer.TotalError();
          double throttle = pidthrottle.TotalError();
          
          // DEBUG
          if (enableTwittle) {
            cout << ccount << " CTE: " << cte << " Steering Value: " << steer_value << endl;
          }
          else {
            cout << "CTE: " << cte << " Steering Value: " << steer_value << endl;
          }

          json msgJson;
          msgJson["steering_angle"] = steer_value;
          msgJson["throttle"] = throttle;
          auto msg = "42[\"steer\"," + msgJson.dump() + "]";
          cout << msg << endl;
          ws.send(msg.data(), msg.length(), uWS::OpCode::TEXT);

          // Check for Twittle
          if (enableTwittle) {
            ccount++;
            if (abs(cte) > 3.0) {
              // off course!  calculate remaining error and cut short
              twittle_err += (cte*cte)*(ncount-ccount);
              cout << "BESTERROR: " << twittle_err << endl;
              exit(EXIT_FAILURE);
            }
            else if (ccount < ncount) {
              twittle_err += cte*cte;
            }
            else {
              cout << "BESTERROR: " << twittle_err << endl;
              exit(EXIT_FAILURE);
            }
          }
        }
      } else {
        // Manual driving
        string msg = "42[\"manual\",{}]";
        ws.send(msg.data(), msg.length(), uWS::OpCode::TEXT);
      }
    }
  });

  // We don't need this since we're not using HTTP but if it's removed the program
  // doesn't compile :-(
  h.onHttpRequest([](uWS::HttpResponse *res, uWS::HttpRequest req, char *data, size_t, size_t) {
    const string s = "<h1>Hello world!</h1>";
    if (req.getUrl().valueLength == 1)
    {
      res->end(s.data(), s.length());
    }
    else
    {
      // i guess this should be done more gracefully?
      res->end(nullptr, 0);
    }
  });

  h.onConnection([&h](uWS::WebSocket<uWS::SERVER> ws, uWS::HttpRequest req) {
    cout << "Connected!!!" << endl;
  });

  h.onDisconnection([&h](uWS::WebSocket<uWS::SERVER> ws, int code, char *message, size_t length) {
    ws.close();
    cout << "Disconnected" << endl;
  });

  int port = 4567;
  if (h.listen(port))
  {
    cout << "Listening to port " << port << endl;
  }
  else
  {
    cerr << "Failed to listen to port" << endl;
    return -1;
  }
  h.run();
}
