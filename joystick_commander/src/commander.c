/**
 * @file commander.c
 * @brief Commander Interface Source.
 *
 */




#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include "polysync_core.h"
#include "polysync_message.h"
#include "polysync_node.h"
#include "polysync_dynamic_driver_commands.h"

#include "joystick.h"
#include "messages.h"
#include "commander.h"




// *****************************************************
// static global types/macros
// *****************************************************

/**
 * @brief Throttle axis index.
 *
 */
#define JSTICK_AXIS_THROTTLE (5)


/**
 * @brief Brake axis index.
 *
 */
#define JSTICK_AXIS_BRAKE (2)


/**
 * @brief Steering axis index.
 *
 */
#define JSTICK_AXIS_STEER (3)


/**
 * @brief Left turn signal button index.
 *
 */
#define JSTICK_BUTTON_LEFT_TURN_SIGNAL (4)


/**
 * @brief Right turn signal button index.
 *
 */
#define JSTICK_BUTTON_RIGHT_TURN_SIGNAL (5)


/**
 * @brief Enable controls button index.
 *
 */
#define JSTICK_BUTTON_ENABLE_CONTROLS (7)

/**
 * @brief Disable controls button index.
 *
 */
#define JSTICK_BUTTON_DISABLE_CONTROLS (6)


/**
 * @brief Shift to park gear button index.
 *
 */
#define JSTICK_BUTTON_GEAR_SHIFT_PARK (3)


/**
 * @brief Shift to drive gear button index.
 *
 */
#define JSTICK_BUTTON_GEAR_SHIFT_DRIVE (0)


/**
 * @brief Shift to neutral gear button index.
 *
 */
#define JSTICK_BUTTON_GEAR_SHIFT_NEUTRAL (2)


/**
 * @brief Shift to reverse gear button index.
 *
 */
#define JSTICK_BUTTON_GEAR_SHIFT_REVERSE (1)


/**
 * @brief Convert degrees to radians.
 *
 */
#define m_radians(deg) ((deg)*(M_PI/180.0))


/**
 * @brief Convert radians to degrees.
 *
 */
#define m_degrees(rad) ((rad)*(180.0/M_PI))


/**
 * @brief Absolute value.
 *
 */
#define m_abs(x) ((x)>0?(x):-(x))




// *****************************************************
// static global data
// *****************************************************

//
static const char *GEAR_POSITION_STRINGS[] =
{
    [GEAR_POSITION_INVALID]         = "INVALID",
    [GEAR_POSITION_UNKNOWN]         = "UNKNOWN",
    [GEAR_POSITION_NOT_AVAILABLE]   = "NOT_AVAILABLE",
    [GEAR_POSITION_PARK]            = "PARK",
    [GEAR_POSITION_REVERSE]         = "REVERSE",
    [GEAR_POSITION_NEUTRAL]         = "NEUTRAL",
    [GEAR_POSITION_DRIVE]           = "DRIVE",
    [GEAR_POSITION_LOW]             = "LOW",
    [GEAR_POSITION_KIND_COUNT]      = "INVALID",
    NULL
};


//
static const char *TURN_SIGNAL_STRINGS[] =
{
    [PLATFORM_TURN_SIGNAL_INVALID]          = "INVALID",
    [PLATFORM_TURN_SIGNAL_UNKNOWN]          = "UNKNOWN",
    [PLATFORM_TURN_SIGNAL_NOT_AVAILABLE]    = "NOT_AVAILABLE",
    [PLATFORM_TURN_SIGNAL_NONE]             = "NONE",
    [PLATFORM_TURN_SIGNAL_LEFT]             = "LEFT",
    [PLATFORM_TURN_SIGNAL_RIGHT]            = "RIGHT",
    [PLATFORM_TURN_SIGNAL_KIND_COUNT]       = "INVALID",
    NULL
};




// *****************************************************
// static declarations
// *****************************************************

//
static int get_brake_setpoint(
        joystick_device_s * const jstick,
        double * const brake );


//
static int get_throttle_setpoint(
        joystick_device_s * const jstick,
        double * const throttle );


//
static int get_steering_setpoint(
        joystick_device_s * const jstick,
        double * const angle );


//
static int get_gear_position(
        joystick_device_s * const jstick,
        ps_gear_position_kind * const gear );


//
static int get_disable_button(
        joystick_device_s * const jstick,
        unsigned int * const state );


//
static int get_enable_button(
        joystick_device_s * const jstick,
        unsigned int * const state );


//
static int publish_brake_command(
        ps_node_ref node_ref,
        joystick_device_s * const jstick,
        ps_platform_brake_command_msg * const msg );


//
static int publish_throttle_command(
        ps_node_ref node_ref,
        joystick_device_s * const jstick,
        ps_platform_throttle_command_msg * const msg );


//
static int publish_steering_command(
        ps_node_ref node_ref,
        joystick_device_s * const jstick,
        ps_platform_steering_command_msg * const msg );


//
static int publish_gear_command(
        ps_node_ref node_ref,
        joystick_device_s * const jstick,
        ps_platform_gear_command_msg * const msg );


//
static int publish_turn_signal_command(
        ps_node_ref node_ref,
        joystick_device_s * const jstick,
        ps_platform_turn_signal_command_msg * const msg );




// *****************************************************
// static definitions
// *****************************************************

//
static int get_brake_setpoint(
        joystick_device_s * const jstick,
        double * const brake )
{
    int ret = DTC_NONE;
    int axis_position = 0;


    // read axis position
    ret = jstick_get_axis(
            jstick,
            JSTICK_AXIS_BRAKE,
            &axis_position );

    if( ret == DTC_NONE )
    {
        // set brake set point - scale to 0:max
        (*brake) = jstick_normalize_axis_position(
                axis_position,
                0.0,
                MAX_BRAKE_PEDAL );
    }


    return ret;
}


//
static int get_throttle_setpoint(
        joystick_device_s * const jstick,
        double * const throttle )
{
    int ret = DTC_NONE;
    int axis_position = 0;


    // read axis position
    ret = jstick_get_axis(
            jstick,
            JSTICK_AXIS_THROTTLE,
            &axis_position );

    // if succeeded
    if( ret == DTC_NONE )
    {
        // set throttle set point - scale to 0:max
        (*throttle) = jstick_normalize_axis_position(
                axis_position,
                0.0,
                MAX_THROTTLE_PEDAL );
    }


    return ret;
}


//
static int get_steering_setpoint(
        joystick_device_s * const jstick,
        double * const angle )
{
    int ret = DTC_NONE;
    int axis_position = 0;


    ret = jstick_get_axis(
            jstick,
            JSTICK_AXIS_STEER,
            &axis_position );

    if( ret == DTC_NONE )
    {
        // set steering wheel angle set point - scale to max:min
        // note that this is inverting the sign of the joystick axis
        (*angle) = jstick_normalize_axis_position(
                axis_position,
                MAX_STEERING_WHEEL_ANGLE,
                MIN_STEERING_WHEEL_ANGLE );
    }


    return ret;
}


//
static int get_gear_position(
        joystick_device_s * const jstick,
        ps_gear_position_kind * const gear )
{
    int ret = DTC_NONE;
    unsigned int btn_state = JOYSTICK_BUTTON_STATE_NOT_PRESSED;


    // default state is disabled meaning we don't publish anything
    (*gear) = GEAR_POSITION_INVALID;

    // read shift-to-reverse button state
    if( ret == DTC_NONE )
    {
        ret = jstick_get_button(
                jstick,
                JSTICK_BUTTON_GEAR_SHIFT_REVERSE,
                &btn_state );
    }

    if( ret == DTC_NONE )
    {
        if( btn_state == JOYSTICK_BUTTON_STATE_PRESSED )
        {
            (*gear) = GEAR_POSITION_REVERSE;
        }
    }

    // read shift-to-drive button state
    if( ret == DTC_NONE )
    {
        ret = jstick_get_button(
                jstick,
                JSTICK_BUTTON_GEAR_SHIFT_DRIVE,
                &btn_state );
    }

    if( ret == DTC_NONE )
    {
        if( btn_state == JOYSTICK_BUTTON_STATE_PRESSED )
        {
            (*gear) = GEAR_POSITION_DRIVE;
        }
    }

    // read shift-to-neutral button state
    if( ret == DTC_NONE )
    {
        ret = jstick_get_button(
                jstick,
                JSTICK_BUTTON_GEAR_SHIFT_NEUTRAL,
                &btn_state );
    }

    if( ret == DTC_NONE )
    {
        if( btn_state == JOYSTICK_BUTTON_STATE_PRESSED )
        {
            (*gear) = GEAR_POSITION_NEUTRAL;
        }
    }

    // read shift-to-park button state
    if( ret == DTC_NONE )
    {
        ret = jstick_get_button(
                jstick,
                JSTICK_BUTTON_GEAR_SHIFT_PARK,
                &btn_state );
    }

    if( ret == DTC_NONE )
    {
        if( btn_state == JOYSTICK_BUTTON_STATE_PRESSED )
        {
            (*gear) = GEAR_POSITION_PARK;
        }
    }


    return ret;
}


//
static int get_turn_signal(
        joystick_device_s * const jstick,
        ps_platform_turn_signal_kind * const turn_signal )
{
    int ret = DTC_NONE;
    unsigned int btn_state = JOYSTICK_BUTTON_STATE_NOT_PRESSED;


    // default state is disabled meaning we don't publish anything
    (*turn_signal) = PLATFORM_TURN_SIGNAL_INVALID;

    // read left-signal button state
    if( ret == DTC_NONE )
    {
        ret = jstick_get_button(
                jstick,
                JSTICK_BUTTON_LEFT_TURN_SIGNAL,
                &btn_state );
    }

    if( ret == DTC_NONE )
    {
        if( btn_state == JOYSTICK_BUTTON_STATE_PRESSED )
        {
            (*turn_signal) = PLATFORM_TURN_SIGNAL_LEFT;
        }
    }

    // read right-signal button state
    if( ret == DTC_NONE )
    {
        ret = jstick_get_button(
                jstick,
                JSTICK_BUTTON_RIGHT_TURN_SIGNAL,
                &btn_state );
    }

    if( ret == DTC_NONE )
    {
        if( btn_state == JOYSTICK_BUTTON_STATE_PRESSED )
        {
            (*turn_signal) = PLATFORM_TURN_SIGNAL_RIGHT;
        }
    }


    return ret;
}


//
static int get_disable_button(
        joystick_device_s * const jstick,
        unsigned int * const state )
{
    int ret = DTC_NONE;
    unsigned int btn_state = JOYSTICK_BUTTON_STATE_NOT_PRESSED;


    // default state is disabled/not-pressed
    (*state) = 0;

    ret = jstick_get_button(
            jstick,
            JSTICK_BUTTON_DISABLE_CONTROLS,
            &btn_state );

    if( ret == DTC_NONE )
    {
        if( btn_state == JOYSTICK_BUTTON_STATE_PRESSED )
        {
            (*state) = 1;
        }
    }


    return ret;
}


//
static int get_enable_button(
        joystick_device_s * const jstick,
        unsigned int * const state )
{
    int ret = DTC_NONE;
    unsigned int btn_state = JOYSTICK_BUTTON_STATE_NOT_PRESSED;


    // default state is disabled/not-pressed
    (*state) = 0;

    ret = jstick_get_button(
            jstick,
            JSTICK_BUTTON_ENABLE_CONTROLS,
            &btn_state );

    if( ret == DTC_NONE )
    {
        if( btn_state == JOYSTICK_BUTTON_STATE_PRESSED )
        {
            (*state) = 1;
        }
    }


    return ret;
}


//
static int publish_brake_command(
        ps_node_ref node_ref,
        joystick_device_s * const jstick,
        ps_platform_brake_command_msg * const msg )
{
    int ret = DTC_NONE;
    double brake_setpoint = 0.0;


    ret = get_brake_setpoint(
            jstick,
            &brake_setpoint );

    if( ret == DTC_NONE )
    {
        msg->brake_command = (DDS_float) brake_setpoint;
        msg->enabled = 1;
        msg->brake_command_type = BRAKE_COMMAND_PEDAL;

        if( brake_setpoint >= BRAKES_ENABLED_MIN )
        {
            // enable on-off indicator
            msg->boo_enabled = 1;
        }
    }

    if( ret == DTC_NONE )
    {
        ret = psync_get_timestamp(
                &msg->header.timestamp );
    }

    if( ret == DTC_NONE )
    {
        ret = psync_message_publish(
                node_ref,
                (ps_msg_ref) msg );
    }


    return ret;
}


//
static int publish_throttle_command(
        ps_node_ref node_ref,
        joystick_device_s * const jstick,
        ps_platform_throttle_command_msg * const msg )
{
    int ret = DTC_NONE;
    double throttle_setpoint = 0.0;
    double brake_setpoint = 0.0;


    ret = get_throttle_setpoint(
            jstick,
            &throttle_setpoint );

    // don't allow throttle if brakes are applied
    if( ret == DTC_NONE )
    {
        ret = get_brake_setpoint(
            jstick,
            &brake_setpoint );

        if( brake_setpoint >= BRAKES_ENABLED_MIN )
        {
            throttle_setpoint = 0.0;
        }
    }

    if( ret == DTC_NONE )
    {
        msg->throttle_command = (DDS_float) throttle_setpoint;
        msg->enabled = 1;
        msg->throttle_command_type = THROTTLE_COMMAND_PEDAL;
    }

    if( ret == DTC_NONE )
    {
        ret = psync_get_timestamp(
                &msg->header.timestamp );
    }

    if( ret == DTC_NONE )
    {
        ret = psync_message_publish(
                node_ref,
                (ps_msg_ref) msg );
    }


    return ret;
}


//
static int publish_steering_command(
        ps_node_ref node_ref,
        joystick_device_s * const jstick,
        ps_platform_steering_command_msg * const msg )
{
    int ret = DTC_NONE;
    double steering_setpoint = 0.0;


    ret = get_steering_setpoint(
            jstick,
            &steering_setpoint );

    if( ret == DTC_NONE )
    {
        msg->steering_wheel_angle = (DDS_float) steering_setpoint;
        msg->enabled = 1;
        msg->steering_command_kind = STEERING_COMMAND_ANGLE;

        // steering rate limit
        msg->max_steering_wheel_rotation_rate = (DDS_float) STEERING_WHEEL_ANGLE_RATE_LIMIT;
    }

    if( ret == DTC_NONE )
    {
        ret = psync_get_timestamp(
                &msg->header.timestamp );
    }

    if( ret == DTC_NONE )
    {
        ret = psync_message_publish(
                node_ref,
                (ps_msg_ref) msg );
    }


    return ret;
}


//
static int publish_gear_command(
        ps_node_ref node_ref,
        joystick_device_s * const jstick,
        ps_platform_gear_command_msg * const msg )
{
    int ret = DTC_NONE;
    ps_gear_position_kind gear_position = GEAR_POSITION_INVALID;


    ret = get_gear_position(
            jstick,
            &gear_position );

    if( ret == DTC_NONE )
    {
        msg->gear_position = gear_position;
    }

    if( ret == DTC_NONE )
    {
        ret = psync_get_timestamp(
                &msg->header.timestamp );
    }

    // only publish if not invalid
    if( ret == DTC_NONE )
    {
        if( gear_position != GEAR_POSITION_INVALID )
        {
            psync_log_debug( "sending gear shift command - %s",
                    GEAR_POSITION_STRINGS[gear_position] );

            ret = psync_message_publish(
                    node_ref,
                    (ps_msg_ref) msg );
        }
    }


    return ret;
}


//
static int publish_turn_signal_command(
        ps_node_ref node_ref,
        joystick_device_s * const jstick,
        ps_platform_turn_signal_command_msg * const msg )
{
    int ret = DTC_NONE;
    ps_platform_turn_signal_kind turn_signal = PLATFORM_TURN_SIGNAL_INVALID;


    ret = get_turn_signal(
            jstick,
            &turn_signal );

    if( ret == DTC_NONE )
    {
        msg->turn_signal = turn_signal;
    }

    if( ret == DTC_NONE )
    {
        ret = psync_get_timestamp(
                &msg->header.timestamp );
    }

    // only publish if not invalid
    if( ret == DTC_NONE )
    {
        if( turn_signal != PLATFORM_TURN_SIGNAL_INVALID )
        {
            psync_log_debug( "sending turn signal command - %s",
                    TURN_SIGNAL_STRINGS[turn_signal] );

            ret = psync_message_publish(
                    node_ref,
                    (ps_msg_ref) msg );
        }
    }


    return ret;
}




// *****************************************************
// public definitions
// *****************************************************

//
int commander_check_for_safe_joystick(
        commander_s * const commander )
{
    int ret = DTC_NONE;
    double brake_setpoint = 0.0;
    double throttle_setpoint = 0.0;


    if( commander == NULL )
    {
        ret = DTC_USAGE;
    }
    else
    {
        // update joystick readings
        ret = jstick_update( &commander->joystick );

        // get brake set point
        ret |= get_brake_setpoint(
                &commander->joystick,
                &brake_setpoint );

        // get throttle set point
        ret |= get_throttle_setpoint(
                &commander->joystick,
                &throttle_setpoint );

        // handle DTC
        if( ret != DTC_NONE )
        {
            // configuration error
            ret = DTC_CONFIG;
        }

        // if succeeded
        if( ret == DTC_NONE )
        {
            // if throttle not zero
            if( throttle_setpoint > 0.0 )
            {
                // invalidate
                ret = DTC_UNAVAILABLE;
            }

            // if brake not zero
            if( brake_setpoint > 0.0 )
            {
                // invalidate
                ret = DTC_UNAVAILABLE;
            }
        }
    }


    return ret;
}


//
int commander_is_valid(
        commander_s * const commander )
{
    int ret = DTC_NONE;


    if( commander == NULL )
    {
        ret = DTC_USAGE;
    }
    else
    {
        ret = messages_is_valid( &commander->messages );
    }


    return ret;
}


//
int commander_set_safe(
        commander_s * const commander )
{
    int ret = DTC_NONE;


    if( commander == NULL )
    {
        ret = DTC_USAGE;
    }
    else
    {
        ret = commander_is_valid( commander );

        if( ret == DTC_NONE )
        {
            ret = messages_set_default_values(
                    commander->dest_control_node_guid,
                    &commander->messages );
        }
    }


    return ret;
}


//
int commander_enumerate_control_nodes(
        ps_node_ref node_ref,
        commander_s * const commander )
{
    int ret = DTC_NONE;


    if( commander == NULL )
    {
        ret = DTC_USAGE;
    }
    else
    {
        // safe state
        ret = commander_set_safe( commander );

        if( ret == DTC_NONE )
        {
            // command ID: enumerate nodes supporting the DataSpeed MKZ sensor interface
            commander->messages.command_msg->id = COMMAND_ID_ENUMERATE_DYNAMIC_DRIVER_INTERFACE_BY_SENSOR;

            // data element zero is the sensor type we want to enumerate
            commander->messages.command_msg->data._buffer[0]._d = PARAMETER_VALUE_ULONGLONG;
            commander->messages.command_msg->data._buffer[0]._u.ull_value =
                    (DDS_unsigned_long_long) SENSOR_TYPE_DATASPEED_MKZ;
            commander->messages.command_msg->data._length = 1;

            // targeted to all nodes
            commander->messages.command_msg->dest_guid = PSYNC_GUID_INVALID;
            ret = psync_guid_set_node_type(
                    PSYNC_NODE_TYPE_ALL,
                    &commander->messages.command_msg->dest_guid );
        }

        // update publish timestamp
        if( ret == DTC_NONE )
        {
            ret = psync_get_timestamp(
                    &commander->messages.command_msg->header.timestamp );
        }

        // publish command
        if( ret == DTC_NONE )
        {
            ret = psync_message_publish(
                    node_ref,
                    commander->messages.command_msg );
        }
    }


    return ret;
}


//
int commander_disable_controls(
        ps_node_ref node_ref,
        commander_s * const commander )
{
    int ret = DTC_NONE;


    psync_log_debug( "sending command to disable controls" );

    if( commander == NULL )
    {
        ret = DTC_USAGE;
    }
    else
    {
        // safe state
        ret = commander_set_safe( commander );

        if( ret == DTC_NONE )
        {
            // command ID: disable controls
            commander->messages.command_msg->id = DISABLE_CONTROLS_COMMAND_ID;

            // set destination node GUID
            commander->messages.command_msg->dest_guid = commander->dest_control_node_guid;

            // update publish timestamp
            ret = psync_get_timestamp(
                    &commander->messages.command_msg->header.timestamp );
        }

        // publish command
        if( ret == DTC_NONE )
        {
            ret = psync_message_publish(
                    node_ref,
                    commander->messages.command_msg );
        }
    }


    return ret;
}


//
int commander_enable_controls(
        ps_node_ref node_ref,
        commander_s * const commander )
{
    int ret = DTC_NONE;


    psync_log_debug( "sending command to enable controls" );

    if( commander == NULL )
    {
        ret = DTC_USAGE;
    }
    else
    {
        // safe state
        ret = commander_set_safe( commander );

        if( ret == DTC_NONE )
        {
            // command ID: disable controls
            commander->messages.command_msg->id = ENABLE_CONTROLS_COMMAND_ID;

            // set destination node GUID
            commander->messages.command_msg->dest_guid = commander->dest_control_node_guid;

            // update publish timestamp
            ret = psync_get_timestamp(
                    &commander->messages.command_msg->header.timestamp );
        }

        // publish command
        if( ret == DTC_NONE )
        {
            ret = psync_message_publish(
                    node_ref,
                    commander->messages.command_msg );
        }
    }


    return ret;
}


//
int commander_estop(
        ps_node_ref node_ref,
        commander_s * const commander )
{
    int ret = DTC_NONE;


    psync_log_debug( "sending e-stop messages" );

    if( commander == NULL )
    {
        ret = DTC_USAGE;
    }
    else
    {
        // safe state
        ret = commander_set_safe( commander );

        if( ret == DTC_NONE )
        {
            // set e-stop
            commander->messages.brake_cmd->e_stop = 1;
            commander->messages.throttle_cmd->e_stop = 1;
            commander->messages.steering_cmd->e_stop = 1;

            // set last known destination GUID
            commander->messages.brake_cmd->dest_guid = commander->dest_control_node_guid;
            commander->messages.throttle_cmd->dest_guid = commander->dest_control_node_guid;
            commander->messages.steering_cmd->dest_guid = commander->dest_control_node_guid;

            // update publish timestamp
            ret = psync_get_timestamp(
                    &commander->messages.command_msg->header.timestamp );
        }

        // publish brake/throttle/steering commands with e-stop enabled
        if( ret == DTC_NONE )
        {
            ret = psync_message_publish(
                    node_ref,
                    (ps_msg_ref) commander->messages.brake_cmd );
        }

        if( ret == DTC_NONE )
        {
            ret = psync_message_publish(
                    node_ref,
                    (ps_msg_ref) commander->messages.throttle_cmd );
        }

        if( ret == DTC_NONE )
        {
            ret = psync_message_publish(
                    node_ref,
                    (ps_msg_ref) commander->messages.steering_cmd );
        }
    }


    return ret;
}


//
int commander_check_reenumeration(
        ps_node_ref node_ref,
        commander_s * const commander )
{
    int ret = DTC_NONE;
    unsigned int disable_button_pressed = 0;


    if( commander == NULL )
    {
        ret = DTC_USAGE;
    }

    // safe state
    if( ret == DTC_NONE )
    {
        ret = commander_set_safe( commander );
    }

    // get 'disable-controls' button state
    if( ret == DTC_NONE )
    {
        ret = get_disable_button(
                &commander->joystick,
                &disable_button_pressed );
    }

    // update joystick
    if( ret == DTC_NONE )
    {
        ret = jstick_update( &commander->joystick );
    }

    // send command to enumerate nodes
    if( ret == DTC_NONE )
    {
        if( disable_button_pressed != 0 )
        {
            psync_log_debug( "sending command to enumerate control nodes" );

            ret = commander_enumerate_control_nodes(
                    node_ref,
                    commander );
        }
    }


    return ret;
}


//
int commander_update(
        ps_node_ref node_ref,
        commander_s * const commander )
{
    int ret = DTC_NONE;
    unsigned int disable_button_pressed = 0;
    unsigned int enable_button_pressed = 0;


    if( commander == NULL )
    {
        ret = DTC_USAGE;
    }

    // safe state
    if( ret == DTC_NONE )
    {
        ret = commander_set_safe( commander );
    }

    // update joystick
    if( ret == DTC_NONE )
    {
        ret = jstick_update( &commander->joystick );
    }

    // get 'disable-controls' button state
    if( ret == DTC_NONE )
    {
        ret = get_disable_button(
                &commander->joystick,
                &disable_button_pressed );
    }

    // get 'enable-controls' button state
    if( ret == DTC_NONE )
    {
        ret = get_enable_button(
                &commander->joystick,
                &enable_button_pressed );
    }

    // only disable if both enable and disable buttons are pressed
    if( (enable_button_pressed != 0) && (disable_button_pressed != 0) )
    {
        enable_button_pressed = 0;
        disable_button_pressed = 1;
    }

    // send command if a enable/disable command
    if( disable_button_pressed != 0 )
    {
        ret = commander_disable_controls(
                node_ref,
                commander );
    }
    else if( enable_button_pressed != 0 )
    {
        ret = commander_enable_controls(
                node_ref,
                commander );
    }
    else
    {
        // publish brake command continously
        if( ret == DTC_NONE )
        {
            ret = publish_brake_command(
                    node_ref,
                    &commander->joystick,
                    commander->messages.brake_cmd );
        }

        // publish throttle command continously
        if( ret == DTC_NONE )
        {
            ret = publish_throttle_command(
                    node_ref,
                    &commander->joystick,
                    commander->messages.throttle_cmd );
        }

        // publish steering command continously
        if( ret == DTC_NONE )
        {
            ret = publish_steering_command(
                    node_ref,
                    &commander->joystick,
                    commander->messages.steering_cmd );
        }

        // publish gear command on button event
        if( ret == DTC_NONE )
        {
            ret = publish_gear_command(
                    node_ref,
                    &commander->joystick,
                    commander->messages.gear_cmd );
        }

        // publish turn signal command on button event
        if( ret == DTC_NONE )
        {
            ret = publish_turn_signal_command(
                    node_ref,
                    &commander->joystick,
                    commander->messages.turn_signal_cmd );
        }
    }


    return ret;
}
