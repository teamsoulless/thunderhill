/*
 * Copyright (c) 2016 PolySync
 *
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */

/**
 * \example HelloWorldSubscriber.cpp
 *
 * PolySync Hello World Subscriber C++ API example application
 *      Demonstrate how to subscribe a node to a message
 *
 */

#include <iostream>
#include <PolySyncNode.hpp>
#include <PolySyncDataModel.hpp>
#include <fstream>
#include <stdio.h>
#include <string>
#include <math.h>
#include <map>

#define PI 3.14159265

using namespace std;

/**
 * @brief HellowWorldSubscriberNode class
 *
 * The HelloWorldSubscriberNode class exists to override the functions defined
 * in the base Node class.  The functions exist in the base class but are
 * stubbed out and must be overloaded in order for them to do something.  In
 * this instance the initStateEvent and the messageHandlerEvent are overloaded
 * to register for the messages and receive them, respectively.
 */
class RecorderNode : public polysync::Node
{

private:
    ps_msg_type _messageType;
    ps_msg_type _imageType;
    ps_msg_type _throttle;
    ps_msg_type _brake;
    ps_msg_type _steer;
    ofstream platformFile = ofstream("platform.csv", std::ios_base::app | std::ofstream::out);
    ofstream commonCSV = ofstream("driving_data.csv", std::ios_base::app | std::ofstream::out);

    char keys[4] = {'B', 'S', 'T', 'I'};

    std::map <char, std::shared_ptr< polysync::Message >> cache;

    void cacheMessage(std::shared_ptr< polysync::Message > message, char key)
    {
        //test if map is full
        if (cache.size() < 4) {
            //find type
            std::map<char, std::shared_ptr< polysync::Message >>::iterator it = cache.find(key);
            if (it != cache.end()) {
                //already there
            } else {
                //add
                cache.insert ( std::pair<char,std::shared_ptr< polysync::Message >>(key, message) );
            }
        } else {
            //map is full, can write message and empty map
            std::string csvString = "";
            std::string tempArray[5] = {"0", "0", "0", "0"};
            for (char key : keys) {
                switch (key) {
                    case 'B':
                        //this fires an error
                        if (std::shared_ptr <PlatformBrakeCommandMessage> incoming = getSubclass< PlatformBrakeCommandMessage >( message )) {
                            insertValue(tempArray, 1, std::to_string(incoming->getBrakeCommand()), 5);
                        };
                        break;
                    case 'S':
                        std::shared_ptr <PlatformSteeringCommandMessage> message = getSubclass< PlatformSteeringCommandMessage >( message );
                        insertValue(tempArray, 2, std::to_string(message->getSteeringWheelAngle()), 5);
                        break;
                    case 'T':
                        std::shared_ptr <PlatformThrottleCommandMessage> message = getSubclass< PlatformThrottleCommandMessage >( message );
                        insertValue(tempArray, 3, std::to_string(message->getThrottleCommand()), 5);
                        break;
                    case 'I':
                        std::shared_ptr <ImageDataMessage> message = getSubclass< ImageDataMessage >( message );
                        insertValue(tempArray, 0, std::to_string(message->getHeaderTimestamp()), 5);
                        break;
                }
            }
        }
    };

    int* insertValue (int* originalArray, int positionToInsertAt, int ValueToInsert, int sizeOfOriginalArray)
    {
      // Create the new array - user must be told to delete it at some point
        int* newArray = new int[sizeOfOriginalArray + 1];
        for (int i=0; i<=sizeOfOriginalArray; ++i)
          {
            if (i < positionToInsertAt)  // All the elements before the one that must be inserted
            {
               newArray[i] = originalArray[i];
            }

            if (i == positionToInsertAt)  // The right place to insert the new element
            {
              newArray[i] = ValueToInsert;
            }

            if (i > positionToInsertAt)  // Now all the remaining elements
            {
              newArray[i] = originalArray[i-1];
            }
          }
        return newArray;
    }



public:
    /**
     * @brief initStateEvent
     *
     * Subscribe to a message that the publisher node will send.
     *
     * @param void
     * @return void
     */
    void initStateEvent() override
    {
        _messageType = getMessageTypeByName( "ps_platform_motion_msg" );
        _imageType = getMessageTypeByName("ps_image_data_msg");

        _brake = getMessageTypeByName("ps_platform_brake_command_msg");
        _steer = getMessageTypeByName("ps_platform_steering_command_msg");
        _throttle = getMessageTypeByName("ps_platform_throttle_command_msg");

        // Register as a listener for the message type that the publisher
        // is going to send.  Message types are defined in later tutorials.
        registerListener( _messageType );
        registerListener(_imageType);
        registerListener(_brake);
        registerListener(_throttle);
        registerListener(_steer);
    }

    /**
     * @brief messageEvent
     *
     * Extract the information from the provided message
     *
     * @param std::shared_ptr< Message > - variable containing the message
     * @return void
     */
    virtual void messageEvent( std::shared_ptr< polysync::Message > message )
    {
        using namespace polysync::datamodel;

        if( std::shared_ptr <PlatformMotionMessage> incomingMessage = getSubclass< PlatformMotionMessage >( message ) )
        {
            DDS_unsigned_long_long ts = incomingMessage->getHeaderTimestamp();

            //to find yaw, use orientation quaternion
            //yaw   =  Mathf.Asin(2*x*y + 2*z*w);
            std::array< DDS_double, 4 > orient = incomingMessage->getOrientation();
            double yaw = asin((2 * orient[0] * orient[1]) + (2 * orient[2] * orient[3]))  * 180.0 / PI;

            //to put to csv: ts, yaw, heading, velocity (x, y, z)
            double heading = incomingMessage->getHeading();
            std::array< DDS_double, 3 > vel = incomingMessage->getVelocity();

            std::string dataString = std::to_string(ts)+","+std::to_string(yaw) +","+std::to_string(heading)+","+std::to_string(vel[0])+","+std::to_string(vel[1])+","+std::to_string(vel[2])+"\n";
            platformFile << dataString;
        }

        // if (std::shared_ptr <PlatformBrakeReportMessage> incoming = getSubclass<PlatformBrakeReportMessage>(message))
        // {
        //     DDS_unsigned_long_long ts = incomingMessage->getHeaderTimestamp();
        //     DDS_float command = incomingMessage->getPedalCommand();
        //     std::string dataString = std::to_string(ts)+","+std::to_string(command);
        //     brakeFile << dataString;
        // }

        // if (std::shared_ptr <PlatformBrakeReportMessage> incoming = getSubclass<PlatformBrakeReportMessage>(message))
        // {
        //     DDS_unsigned_long_long ts = incomingMessage->getHeaderTimestamp();
        //     DDS_float command = incomingMessage->getPedalCommand();
        //     std::string dataString = std::to_string(ts)+","+std::to_string(command);
        //     brakeFile << dataString;
        // }


        if (std::shared_ptr < ImageDataMessage > incomingMessage = getSubclass < ImageDataMessage > (message))
        {
            // FILE *stream;
            // DDS_unsigned_long_long ts = incomingMessage->getHeaderTimestamp();
            // std::string imageName = "IMG/" + std::to_string(ts) + ".jpeg";
            // const char * c = imageName.c_str();

            // if((stream = freopen(c, "w", stdout)) == NULL) {
            //     exit(-1);
            // }

            // std::vector < DDS_octet > imageData = incomingMessage->getDataBuffer();
            // for (int d: imageData) {
            //     printf("%c", d);
            // }
        }
    }

};

/**
 * @brief main
 *
 * Entry point for this tutorial application
 * The "connectPolySync" begins this node's PolySync execution loop.
 *
 * @return int - exit code
 */
int main()
{
    // Create an instance of the HelloWorldNode and connect it to PolySync
    RecorderNode recorderNode;

    // When the node has been created, it will cause an initStateEvent to
    // to be sent.
    recorderNode.connectPolySync();

    return 0;
}
