/**
 * @file messages.c
 * @brief Message Utilities Interface Source.
 *
 */




#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <glib-2.0/glib.h>

#include "polysync_core.h"
#include "polysync_message.h"
#include "polysync_node.h"
#include "polysync_dynamic_driver_commands.h"

#include "messages.h"




// *****************************************************
// static global types/macros
// *****************************************************




// *****************************************************
// static global data
// *****************************************************

/**
 * @brief Platform brake command message type name.
 *
 */
static const char BRAKE_COMMAND_MSG_NAME[] = "ps_platform_brake_command_msg";


/**
 * @brief Platform throttle command message type name.
 *
 */
static const char THROTTLE_COMMAND_MSG_NAME[] = "ps_platform_throttle_command_msg";


/**
 * @brief Platform steering command message type name.
 *
 */
static const char STEERING_COMMAND_MSG_NAME[] = "ps_platform_steering_command_msg";


/**
 * @brief Gear position command message type name.
 *
 */
static const char GEAR_POSITION_COMMAND_MSG_NAME[] = "ps_platform_gear_command_msg";


/**
 * @brief Turn signal command message type name.
 *
 */
static const char TURN_SIGNAL_COMMAND_MSG_NAME[] = "ps_platform_turn_signal_command_msg";


/**
 * @brief PolySync command message type name.
 *
 */
static const char COMMAND_MSG_NAME[] = "ps_command_msg";


/**
 * @brief PolySync response message type name.
 *
 */
static const char RESPONSE_MSG_NAME[] = "ps_response_msg";


//
static GMutex global_guid_mutex = G_STATIC_MUTEX_INIT;


//
static ps_guid global_discovered_guid = PSYNC_GUID_INVALID;




// *****************************************************
// static declarations
// *****************************************************

/**
 * @brief Response message handler.
 *
 * Updates the global discovered GUID if valid response is received.
 *
 * @param [in] msg_type Message type received.
 * @param [in] message Reference to the received message.
 * @param [in] user_data Our node reference.
 *
 */
static void ps_response_msg_handler(
        const ps_msg_type msg_type,
        const ps_msg_ref const message,
        void * const user_data );


//
static int get_message_type(
        ps_node_ref node_ref,
        const char * const msg_name,
        ps_msg_type * const msg_type );


//
static int get_message_and_type(
        ps_node_ref node_ref,
        const char * const msg_name,
        ps_msg_type * const msg_type,
        ps_msg_ref * msg_ref_ptr );


//
static int set_default_brake_command(
        const ps_guid dest_guid,
        ps_platform_brake_command_msg * const msg );


//
static int set_default_throttle_command(
        const ps_guid dest_guid,
        ps_platform_throttle_command_msg * const msg );


//
static int set_default_steering_command(
        const ps_guid dest_guid,
        ps_platform_steering_command_msg * const msg );


//
static int set_default_gear_command(
        const ps_guid dest_guid,
        ps_platform_gear_command_msg * const msg );


//
static int set_default_turn_signal_command(
        const ps_guid dest_guid,
        ps_platform_turn_signal_command_msg * const msg );


//
static int set_default_command_msg(
        const ps_guid dest_guid,
        ps_command_msg * const msg );




// *****************************************************
// static definitions
// *****************************************************

//
static void ps_response_msg_handler(
        const ps_msg_type msg_type,
        const ps_msg_ref const message,
        void * const user_data )
{
    if( user_data == NULL )
    {
        return;
    }

    ps_node_ref node_ref = (ps_node_ref) user_data;
    ps_guid my_guid = 0;
    const ps_response_msg * const rsp_msg = (ps_response_msg*) message;


    // get our node's GUID
    (void) psync_node_get_guid( node_ref, &my_guid );

    // ignore our own messages
    if( my_guid == rsp_msg->header.src_guid )
    {
        return;
    }

    // get source GUID if responding to our enumeration command
    if( rsp_msg->id == COMMAND_ID_ENUMERATE_DYNAMIC_DRIVER_INTERFACE_BY_SENSOR )
    {
        if( rsp_msg->dtc == DTC_NONE )
        {
            if( rsp_msg->data._length == 1 )
            {
                if( rsp_msg->data._buffer[0]._u.ull_value == SENSOR_TYPE_DATASPEED_MKZ )
                {
                    g_mutex_lock( &global_guid_mutex );

                    if( global_discovered_guid == PSYNC_GUID_INVALID )
                    {
                        global_discovered_guid = rsp_msg->header.src_guid;
                    }

                    g_mutex_unlock( &global_guid_mutex );
                }
            }
        }
    }
}


//
static int get_message_type(
        ps_node_ref node_ref,
        const char * const msg_name,
        ps_msg_type * const msg_type )
{
    int ret = DTC_NONE;


    // get message type by name
    ret = psync_message_get_type_by_name(
            node_ref,
            msg_name,
            msg_type );

    // error check
    if( ret == DTC_TYPESUPPORT )
    {
        psync_log_error(
                "missing data model support for message type '%s'",
                msg_name );
    }


    return ret;
}


//
static int get_message_and_type(
        ps_node_ref node_ref,
        const char * const msg_name,
        ps_msg_type * const msg_type,
        ps_msg_ref * msg_ref_ptr )
{
    int ret = DTC_NONE;


    // get message type by name using local routine
    ret = get_message_type(
            node_ref,
            msg_name,
            msg_type );

    // allocate message
    if( ret == DTC_NONE )
    {
        ret = psync_message_alloc(
                node_ref,
                (*msg_type),
                msg_ref_ptr );
    }


    return ret;
}


//
static int set_default_brake_command(
        const ps_guid dest_guid,
        ps_platform_brake_command_msg * const msg )
{
    int ret = DTC_NONE;


    if( msg == NULL )
    {
        ret = DTC_USAGE;
    }
    else
    {
        // set destination GUID
        msg->dest_guid = dest_guid;

        // clear e-stop
        msg->e_stop = PSYNC_EMERGENCY_STOP_DISABLED;

        // disable
        msg->enabled = 0;

        // BOO disabled
        msg->boo_enabled = 0;

        // invalidate
        msg->brake_command_type = BRAKE_COMMAND_INVALID;

        // zero
        msg->brake_command = (DDS_float) 0.0f;

        // update timestamp
        ret = psync_get_timestamp( &msg->timestamp );
    }


    return ret;
}


//
static int set_default_throttle_command(
        const ps_guid dest_guid,
        ps_platform_throttle_command_msg * const msg )
{
    int ret = DTC_NONE;


    if( msg == NULL )
    {
        ret = DTC_USAGE;
    }
    else
    {
        // set destination GUID
        msg->dest_guid = dest_guid;

        // clear e-stop
        msg->e_stop = PSYNC_EMERGENCY_STOP_DISABLED;

        // disable
        msg->enabled = 0;

        // invalidate
        msg->throttle_command_type = THROTTLE_COMMAND_INVALID;

        // zero
        msg->throttle_command = (DDS_float) 0.0f;

        // update timestamp
        ret = psync_get_timestamp( &msg->timestamp );
    }


    return ret;
}


//
static int set_default_steering_command(
        const ps_guid dest_guid,
        ps_platform_steering_command_msg * const msg )
{
    int ret = DTC_NONE;


    if( msg == NULL )
    {
        ret = DTC_USAGE;
    }
    else
    {
        // set destination GUID
        msg->dest_guid = dest_guid;

        // clear e-stop
        msg->e_stop = PSYNC_EMERGENCY_STOP_DISABLED;

        // disable
        msg->enabled = 0;

        // invalidate
        msg->steering_command_kind = STEERING_COMMAND_INVALID;

        // zero
        msg->steering_wheel_angle = (DDS_float) 0.0f;

        // zero
        msg->max_steering_wheel_rotation_rate = (DDS_float) 0.0f;

        // update timestamp
        ret = psync_get_timestamp( &msg->timestamp );
    }


    return ret;
}


//
static int set_default_gear_command(
        const ps_guid dest_guid,
        ps_platform_gear_command_msg * const msg )
{
    int ret = DTC_NONE;


    if( msg == NULL )
    {
        ret = DTC_USAGE;
    }
    else
    {
        // set destination GUID
        msg->dest_guid = dest_guid;

        // clear e-stop
        msg->e_stop = PSYNC_EMERGENCY_STOP_DISABLED;

        // invalidate command
        msg->gear_position = GEAR_POSITION_INVALID;

        // update timestamp
        ret = psync_get_timestamp( &msg->timestamp );
    }


    return ret;
}


//
static int set_default_turn_signal_command(
        const ps_guid dest_guid,
        ps_platform_turn_signal_command_msg * const msg )
{
    int ret = DTC_NONE;


    if( msg == NULL )
    {
        ret = DTC_USAGE;
    }
    else
    {
        // set destination GUID
        msg->dest_guid = dest_guid;

        // clear e-stop
        msg->e_stop = PSYNC_EMERGENCY_STOP_DISABLED;

        // invalidate command
        msg->turn_signal = PLATFORM_TURN_SIGNAL_INVALID;

        // update timestamp
        ret = psync_get_timestamp( &msg->timestamp );
    }


    return ret;
}


//
static int set_default_command_msg(
        const ps_guid dest_guid,
        ps_command_msg * const msg )
{
    int ret = DTC_NONE;


    if( msg == NULL )
    {
        ret = DTC_USAGE;
    }
    else
    {
        // set destination GUID
        msg->dest_guid = dest_guid;

        // clear ID
        msg->id = 0;

        // zero data buffer length
        msg->data._length = 0;

        if( msg->data._buffer == NULL )
        {
            msg->data._buffer = DDS_sequence_ps_parameter_value_allocbuf( 1 );

            if( msg->data._buffer == NULL )
            {
                ret = DTC_MEMERR;
            }
            else
            {
                msg->data._maximum = 1;
                msg->data._release = 1;
                memset( msg->data._buffer, 0, sizeof(*msg->data._buffer) * msg->data._maximum );
            }
        }

        // update timestamp
        if( ret == DTC_NONE )
        {
            ret = psync_get_timestamp( &msg->timestamp );
        }
    }


    return ret;
}




// *****************************************************
// public definitions
// *****************************************************

//
int messages_alloc(
        ps_node_ref node_ref,
        messages_s * const messages )
{
    int ret = DTC_NONE;
    ps_msg_type msg_type = PSYNC_MSG_TYPE_INVALID;


    if( messages == NULL )
    {
        ret = DTC_USAGE;
    }
    else
    {
        // get brake command message
        if( ret == DTC_NONE )
        {
            ret = get_message_and_type(
                    node_ref,
                    BRAKE_COMMAND_MSG_NAME,
                    &msg_type,
                    (ps_msg_ref*) &messages->brake_cmd );
        }

        // get throttle command message
        if( ret == DTC_NONE )
        {
            ret = get_message_and_type(
                    node_ref,
                    THROTTLE_COMMAND_MSG_NAME,
                    &msg_type,
                    (ps_msg_ref*) &messages->throttle_cmd );
        }

        // get steering command message
        if( ret == DTC_NONE )
        {
            ret = get_message_and_type(
                    node_ref,
                    STEERING_COMMAND_MSG_NAME,
                    &msg_type,
                    (ps_msg_ref*) &messages->steering_cmd );
        }

        // get gear position command message
        if( ret == DTC_NONE )
        {
            ret = get_message_and_type(
                    node_ref,
                    GEAR_POSITION_COMMAND_MSG_NAME,
                    &msg_type,
                    (ps_msg_ref*) &messages->gear_cmd );
        }

        // get turn signal command message
        if( ret == DTC_NONE )
        {
            ret = get_message_and_type(
                    node_ref,
                    TURN_SIGNAL_COMMAND_MSG_NAME,
                    &msg_type,
                    (ps_msg_ref*) &messages->turn_signal_cmd );
        }

        // get PolySync command message
        if( ret == DTC_NONE )
        {
            ret = get_message_and_type(
                    node_ref,
                    COMMAND_MSG_NAME,
                    &msg_type,
                    (ps_msg_ref*) &messages->command_msg );
        }

        // get PolySync response message type
        if( ret == DTC_NONE )
        {
            ret = get_message_type(
                    node_ref,
                    RESPONSE_MSG_NAME,
                    &messages->response_msg_type );
        }

        // default values
        if( ret == DTC_NONE )
        {
            ret = messages_set_default_values(
                    PSYNC_GUID_INVALID,
                    messages );
        }
    }


    return ret;
}


//
int messages_register_response_subscriber(
        ps_node_ref node_ref,
        messages_s * const messages )
{
    int ret = DTC_NONE;


    if( messages == NULL )
    {
        ret = DTC_USAGE;
    }
    else
    {
        ret = psync_node_set_subscriber_reliability_qos(
                node_ref,
                messages->response_msg_type,
                RELIABILITY_QOS_RELIABLE );

        if( ret == DTC_NONE )
        {
            ret = psync_message_register_listener(
                    node_ref,
                    messages->response_msg_type,
                    ps_response_msg_handler,
                    (void*) node_ref );
        }
    }


    return ret;
}


//
int messages_unregister_response_subscriber(
        ps_node_ref node_ref,
        messages_s * const messages )
{
    int ret = DTC_NONE;


    if( messages == NULL )
    {
        ret = DTC_USAGE;
    }
    else
    {
        ret = psync_message_unregister_listener(
                node_ref,
                messages->response_msg_type );
    }


    return ret;
}


//
int messages_free(
        ps_node_ref node_ref,
        messages_s * const messages )
{
    int ret = DTC_NONE;


    if( messages == NULL )
    {
        ret = DTC_USAGE;
    }
    else
    {
        ret |= psync_message_free(
                node_ref,
                (ps_msg_ref*) &messages->brake_cmd );

        ret |= psync_message_free(
                node_ref,
                (ps_msg_ref*) &messages->throttle_cmd );

        ret |= psync_message_free(
                node_ref,
                (ps_msg_ref*) &messages->steering_cmd );

        ret |= psync_message_free(
                node_ref,
                (ps_msg_ref*) &messages->gear_cmd );

        ret |= psync_message_free(
                node_ref,
                (ps_msg_ref*) &messages->turn_signal_cmd );

        ret |= psync_message_free(
                node_ref,
                (ps_msg_ref*) &messages->command_msg );

        if( ret != DTC_NONE )
        {
            ret = DTC_MEMERR;
        }
    }


    return ret;
}


//
int messages_is_valid(
        const messages_s * const const messages )
{
    int ret = DTC_NONE;


    if( messages == NULL )
    {
        ret = DTC_USAGE;
    }
    else
    {
        if( messages->brake_cmd == NULL )
        {
            ret = DTC_DATAERR;
        }

        if( messages->throttle_cmd == NULL )
        {
            ret = DTC_DATAERR;
        }

        if( messages->steering_cmd == NULL )
        {
            ret = DTC_DATAERR;
        }

        if( messages->gear_cmd == NULL )
        {
            ret = DTC_DATAERR;
        }

        if( messages->turn_signal_cmd == NULL )
        {
            ret = DTC_DATAERR;
        }

        if( messages->command_msg == NULL )
        {
            ret = DTC_DATAERR;
        }

        if( messages->response_msg_type == PSYNC_MSG_TYPE_INVALID )
        {
            ret = DTC_DATAERR;
        }

        if( messages->command_msg->data._buffer == NULL )
        {
            ret = DTC_DATAERR;
        }

        if( messages->command_msg->data._maximum == 0 )
        {
            ret = DTC_DATAERR;
        }
    }


    return ret;
}


//
int messages_set_default_values(
        const ps_guid dest_guid,
        messages_s * const messages )
{
    int ret = DTC_NONE;


    if( messages == NULL )
    {
        ret = DTC_USAGE;
    }
    else
    {
        // default brake command values
        ret |= set_default_brake_command(
                dest_guid,
                messages->brake_cmd );

        // default throttle command values
        ret |= set_default_throttle_command(
                dest_guid,
                messages->throttle_cmd );

        // default steering command values
        ret |= set_default_steering_command(
                dest_guid,
                messages->steering_cmd );

        // default gear command values
        ret |= set_default_gear_command(
                dest_guid,
                messages->gear_cmd );

        // default turn signal command values
        ret |= set_default_turn_signal_command(
                dest_guid,
                messages->turn_signal_cmd );

        // default PolySync command values
        ret |= set_default_command_msg(
                dest_guid,
                messages->command_msg );

        if( ret != DTC_NONE )
        {
            ret = DTC_MEMERR;
        }
    }


    return ret;
}


//
int messages_get_discovered_guid(
        ps_guid * const guid )
{
    int ret = DTC_NONE;


    if( guid == NULL )
    {
        ret = DTC_USAGE;
    }
    else
    {
        g_mutex_lock( &global_guid_mutex );

        (*guid) = global_discovered_guid;

        g_mutex_unlock( &global_guid_mutex );
    }


    return ret;
}


//
int messages_reset_discovered_guid( void )
{
    int ret = DTC_NONE;


    g_mutex_lock( &global_guid_mutex );

    global_discovered_guid = PSYNC_GUID_INVALID;

    g_mutex_unlock( &global_guid_mutex );


    return ret;
}
