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
#include "polysync_dynamic_driver_commands.h"
#include <math.h>

#ifndef NODE_FLAGS_VALUE
#define NODE_FLAGS_VALUE (0)
#endif

#ifndef ENABLE_CONTROLS_COMMAND_ID
#define ENABLE_CONTROLS_COMMAND_ID (5001)
#endif

using namespace std;

typedef void (*imageCallback)(int, unsigned char *, float, float, float);

class MainNode : public polysync::Node
{
private:
    const string node_name = "polysync-python-test-client-cpp";
    ps_msg_type _messageType;
    ps_msg_type _brakeReport;
    ps_msg_type _steeringReport;
    ps_msg_type _throttleReport;
    ps_msg_type _platformMotion;

    std::vector <unsigned char> imageData;
    unsigned int imageSize;

    int maxGen = 1;
    int gen = 0;
    int commandMessageInitialized = 0;

    unsigned long long steeringID = 0;
    unsigned long long brakeID = 0;
    unsigned long long throttleID = 0;

    double velocity = 0;
    double latitude = 0;
    double longitude = 0;

public:
    imageCallback imageRecieved = NULL;

    MainNode()
    {
        setNodeType( PSYNC_NODE_TYPE_API_USER );
        setDomainID( PSYNC_DEFAULT_DOMAIN );
        setSDFID( PSYNC_SDF_ID_INVALID );
        setFlags( NODE_FLAGS_VALUE );
        setNodeName( node_name );
    }

    ~MainNode()
    {

    }

    void initStateEvent() override
    {
        _messageType = getMessageTypeByName( "ps_image_data_msg" );
        registerListener( _messageType );
         _steeringReport = getMessageTypeByName( "ps_platform_steering_report_msg" );
        registerListener( _steeringReport );
         _brakeReport = getMessageTypeByName( "ps_platform_brake_report_msg" );
        registerListener( _brakeReport );
         _throttleReport = getMessageTypeByName( "ps_platform_throttle_report_msg" );
        registerListener( _throttleReport );
        _platformMotion = getMessageTypeByName( "ps_platform_motion_msg");
        registerListener( _platformMotion );
    }

    void okStateEvent() override
    {
        if (commandMessageInitialized == 0 && brakeID != 0)
        {
            polysync::datamodel::CommandMessage cmdMsg( *this);
            cmdMsg.setId(ENABLE_CONTROLS_COMMAND_ID);
            cmdMsg.setDestGuid(brakeID);
            cmdMsg.setHeaderTimestamp( polysync::getTimestamp() );
            cmdMsg.setTimestamp( polysync::getTimestamp() );
            cmdMsg.publish();
            commandMessageInitialized = 1;
        }
    }

    virtual void messageEvent( std::shared_ptr< polysync::Message > message )
    {
        using namespace polysync::datamodel;

        if (std::shared_ptr < PlatformSteeringReportMessage > incomingMessage = getSubclass < PlatformSteeringReportMessage > (message))
        {
            if (steeringID == 0)
            {
                steeringID = incomingMessage->getHeaderSrcGuid();
            }
        }

        if (std::shared_ptr < PlatformBrakeReportMessage > incomingMessage = getSubclass < PlatformBrakeReportMessage > (message))
        {
            if (brakeID == 0)
            {
                brakeID = incomingMessage->getHeaderSrcGuid();
            }
        }

        if (std::shared_ptr < PlatformThrottleReportMessage > incomingMessage = getSubclass < PlatformThrottleReportMessage > (message))
        {
            if (throttleID == 0)
            {
                throttleID = incomingMessage->getHeaderSrcGuid();
            }
        }

        gen++;
        if (gen == maxGen)
        {
            gen = 0;

            if (std::shared_ptr < PlatformMotionMessage > incomingMessage = getSubclass < PlatformMotionMessage > (message))
            {
                std::array< DDS_double, 3 > vel = incomingMessage->getVelocity();

                if (sizeof(vel)/sizeof(vel[0]) == 3)
                {
                    velocity = sqrt(pow(vel[0], 2) + pow(vel[1], 2) + pow(vel[2], 2));
                }
                latitude = incomingMessage->getLatitude();
                longitude = incomingMessage->getLongitude();
                //std::cout << "got speed: " << velocity << " lat: " << latitude << " long: " << longitude << std::endl;
            }

            if (std::shared_ptr < ImageDataMessage > incomingMessage = getSubclass < ImageDataMessage > (message))
            {
                std::cout << "image received" << std::endl;
                std::vector <unsigned char> image = incomingMessage->getDataBuffer();
                imageSize = image.size();
                if (imageSize > imageData.size())
                {
                    imageData = vector<unsigned char> (imageSize);
                }
                unsigned char *p = imageData.data();
                unsigned char *q = image.data();
                for (unsigned int i=0; i<imageSize; i++)
                {
                    *p = *q;
                    ++p;
                    ++q;
                }
                if (imageRecieved != NULL) {
                    imageRecieved(imageData.size(), imageData.data(), velocity, latitude, longitude);
                }
            }

        }

    }

    void steerCommand(float angle)
    {
        if (commandMessageInitialized != 0 && steeringID != 0)
        {
            polysync::datamodel::PlatformSteeringCommandMessage message( *this);
            message.setDestGuid(steeringID);
            message.setTimestamp( polysync::getTimestamp() );
            message.setSteeringWheelAngle(angle);
            message.setMaxSteeringWheelRotationRate(M_PI_2);
            message.setHeaderTimestamp( polysync::getTimestamp() );
            message.setSteeringCommandKind(STEERING_COMMAND_ANGLE);
            message.setEnabled(1);
            message.publish();
            message.print();
        }
    }

    void brakeCommand(float value)
    {
        if (commandMessageInitialized != 0 && brakeID != 0)
        {
            polysync::datamodel::PlatformBrakeCommandMessage message( *this);
            message.setDestGuid(brakeID);
            message.setTimestamp( polysync::getTimestamp() );
            message.setBrakeCommand(value);
            message.setHeaderTimestamp( polysync::getTimestamp() );
            message.setBrakeCommandType(BRAKE_COMMAND_PEDAL);
            message.setEnabled(1);
            message.publish();
            message.print();
        }
    }

    void throttleCommand(float value)
    {
        if (commandMessageInitialized != 0 && brakeID != 0)
        {
            polysync::datamodel::PlatformThrottleCommandMessage message( *this);
            message.setDestGuid(throttleID);
            message.setTimestamp( polysync::getTimestamp() );
            message.setThrottleCommand(value);
            message.setHeaderTimestamp( polysync::getTimestamp() );
            message.setThrottleCommandType(THROTTLE_COMMAND_PEDAL);
            message.setEnabled(1);
            message.publish();
            message.print();
        }
    }

};

extern "C" {
    MainNode* MainNode_new(){ return new MainNode(); }
    void MainNode_connectPolySync(MainNode* node){ node->connectPolySync(); }
    void MainNode_setImageCallback(MainNode* node, imageCallback imageRecieved){ node->imageRecieved = imageRecieved; }
    void MainNode_steerCommand(MainNode* node, float angle){ node->steerCommand(angle); }
    void MainNode_brakeCommand(MainNode* node, float value){ node->brakeCommand(value); }
    void MainNode_throttleCommand(MainNode* node, float value){ node->throttleCommand(value); }
}

// int main()
// {
//     MainNode node;
//     node.connectPolySync();

//     return 0;
// }
